#!/usr/bin/env python3
"""
ikizamini_local_parallel.py

Parallel CLI runner for IKIZAMINI using Ollama (e.g. gemma3).

It processes learning objectives in parallel to reduce wall-clock time and saves
outputs under the folder:
  Parallely_Processed/

This script reuses the core generator/formatting logic from `ikizamini_local.py`.

Examples:
  python3 ikizamini_local_parallel.py --input Uru.txt
  python3 ikizamini_local_parallel.py --input Uru.txt --workers 4
  python3 ikizamini_local_parallel.py --input Uru.txt --worker-model gemma3 --manager-model gemma3

Notes:
- Too much parallelism can overload Ollama/GPU VRAM. Start with 2–4 workers.
- Outputs are written by the parent process to avoid file-write races.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Reuse generator + parsing/formatting from the UI module
from ikizamini_local import (  # type: ignore
    parse_objectives,
    collect_learning_objectives_with_paths,
    generate_for_objective_strict,
    objective_file_text,
    unique_objective_filename,
)


DEFAULT_OUTPUT_DIR = "Parallely_Processed"


@dataclass(frozen=True)
class Objective:
    objective_id: str
    objective_text: str
    path: List[Tuple[str, str]]


@dataclass(frozen=True)
class Result:
    objective_id: str
    objective_text: str
    path: List[Tuple[str, str]]
    ok: bool
    elapsed_s: float
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _process_one(
    base_url: str,
    worker_model: str,
    manager_model: str,
    max_rounds: int,
    num_ctx: int,
    timeout_s: int,
    max_retries: int,
    obj: Objective,
) -> Result:
    start = time.time()
    try:
        out = generate_for_objective_strict(
            base_url=base_url,
            worker_model=worker_model,
            manager_model=manager_model,
            objective_id=obj.objective_id,
            objective_text=obj.objective_text,
            max_rounds=max_rounds,
            num_ctx=num_ctx,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        return Result(
            objective_id=obj.objective_id,
            objective_text=obj.objective_text,
            path=obj.path,
            ok=True,
            elapsed_s=time.time() - start,
            data=out,
        )
    except Exception as e:
        return Result(
            objective_id=obj.objective_id,
            objective_text=obj.objective_text,
            path=obj.path,
            ok=False,
            elapsed_s=time.time() - start,
            error=str(e),
        )


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Parallel IKIZAMINI runner (Ollama).")
    ap.add_argument("--input", required=True, help="Path to objectives outline .txt (e.g. Uru.txt)")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--worker-model", default="gemma3", help="Ollama worker model (default: gemma3)")
    ap.add_argument("--manager-model", default="gemma3", help="Ollama manager model (default: gemma3)")
    ap.add_argument("--max-rounds", type=int, default=6, help="Max repair/review rounds per objective")
    ap.add_argument("--num-ctx", type=int, default=8192, help="Ollama context size (num_ctx)")
    ap.add_argument("--timeout-s", type=int, default=int(os.environ.get("IKIZAMINI_OLLAMA_TIMEOUT", "600")), help="Ollama request timeout seconds")
    ap.add_argument("--max-retries", type=int, default=int(os.environ.get("IKIZAMINI_OLLAMA_RETRIES", "4")), help="Retries per Ollama call")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of learning objectives (0 = all)")
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("IKIZAMINI_PARALLEL_WORKERS", "2")),
        help="Parallel worker processes (default: env IKIZAMINI_PARALLEL_WORKERS or 2)",
    )
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    roots = parse_objectives(content)
    objs_raw = collect_learning_objectives_with_paths(roots)
    if args.limit and args.limit > 0:
        objs_raw = objs_raw[: args.limit]

    objectives: List[Objective] = [
        Objective(o["objective_id"], o["objective_text"], o["path"])
        for o in objs_raw
    ]

    if not objectives:
        print("[OK] No learning objectives found.")
        return 0

    _safe_mkdir(args.output_dir)

    total = len(objectives)
    print(f"[START] Objectives: {total} | workers={args.workers} | model={args.worker_model}/{args.manager_model}")
    print(f"[OUT] {os.path.abspath(args.output_dir)}")

    # Run in parallel
    futures = []
    completed = 0
    ok_count = 0
    fail_count = 0

    # We'll write files as results complete (parent process only)
    index_lines: List[str] = []
    index_lines.append("IKIZAMINI PARALLELY_PROCESSED INDEX (Ollama)")
    index_lines.append(f"Ollama URL: {args.ollama_url}")
    index_lines.append(f"Worker model: {args.worker_model}")
    index_lines.append(f"Manager model: {args.manager_model}")
    index_lines.append(f"Generated at (unix): {int(time.time())}")
    index_lines.append("")

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for obj in objectives:
            futures.append(
                ex.submit(
                    _process_one,
                    args.ollama_url,
                    args.worker_model,
                    args.manager_model,
                    args.max_rounds,
                    args.num_ctx,
                    args.timeout_s,
                    args.max_retries,
                    obj,
                )
            )

        for fut in as_completed(futures):
            res: Result = fut.result()
            completed += 1
            base = unique_objective_filename(res.objective_id, res.objective_text)
            json_path = os.path.join(args.output_dir, f"{base}.json")
            txt_path = os.path.join(args.output_dir, f"{base}.txt")

            if res.ok and res.data is not None:
                ok_count += 1
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(res.data, jf, ensure_ascii=False, indent=2)
                txt = objective_file_text(res.path, res.objective_id, res.objective_text, res.data, None)
                with open(txt_path, "w", encoding="utf-8") as tf:
                    tf.write(txt)
                index_lines.append(f"- {os.path.basename(txt_path)}")
                print(f"[{completed}/{total}] OK   {res.objective_id} ({res.elapsed_s:.1f}s)")
            else:
                fail_count += 1
                err = res.error or "Unknown error"
                txt = objective_file_text(res.path, res.objective_id, res.objective_text, None, err)
                with open(txt_path, "w", encoding="utf-8") as tf:
                    tf.write(txt)
                index_lines.append(f"- {os.path.basename(txt_path)}  (FAILED)")
                print(f"[{completed}/{total}] FAIL {res.objective_id} ({res.elapsed_s:.1f}s): {err[:140]}")

    # Write index
    with open(os.path.join(args.output_dir, "index.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines).rstrip() + "\n")

    print(f"[DONE] Success={ok_count} Failed={fail_count} Total={total}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

