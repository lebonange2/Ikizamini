#!/usr/bin/env python3
"""
ikizamini_local_parallel.py

IKIZAMINI Local Web UI (Flask) + Ollama (PARALLEL objective processing).

This file provides the SAME durable UI as `ikizamini_local.py` (same routes, same
templates, same config fields, same defaults), but the background job runner
processes learning objectives concurrently to reduce wall-clock time.

How it works:
- We import `ikizamini_local` and reuse its `app`, routes, DB schema, templates, etc.
- We override the background `run_job()` implementation to process objectives in parallel.

Tuning:
- Set `IKIZAMINI_PARALLEL_WORKERS` to control objective concurrency per job (default: 4).

Run:
  python3 ikizamini_local_parallel.py
"""

from __future__ import annotations

import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Any, Dict, List, Optional, Tuple

import ikizamini_local as base  # reuse the full UI + persistence
import subprocess


PARALLEL_WORKERS = int(os.environ.get("IKIZAMINI_PARALLEL_WORKERS", "4"))
LOG_LOCK = threading.Lock()


def _cancel_requested(job_id: str) -> bool:
    return base.is_cancel_requested(job_id)  # type: ignore


def _log(job_id: str, message: str) -> None:
    # Avoid interleaving when many objectives finish at once.
    with LOG_LOCK:
        base.append_job_log(job_id, message)


def _update_job_progress(job_id: str, failures: List[str]) -> None:
    conn = base.db_connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) AS n FROM objectives WHERE job_id=? AND status IN ('done','failed')",
        (job_id,),
    )
    done_count = cur.fetchone()["n"]
    cur.execute(
        "UPDATE jobs SET completed=?, failures_json=?, heartbeat_unix=? WHERE job_id=?",
        (done_count, json.dumps(failures, ensure_ascii=False), int(time.time()), job_id),
    )
    conn.commit()
    conn.close()


def _process_objective(
    job_id: str,
    obj: Dict[str, Any],
    idx: int,
    total: int,
    ollama_url: str,
    worker_model: str,
    manager_model: str,
    max_rounds: int,
    num_ctx: int,
    timeout_s: int,
    max_retries: int,
    failures: List[str],
    failures_lock: threading.Lock,
) -> None:
    obj_id = obj["objective_id"]
    obj_text = obj["objective_text"]
    path = obj["path"]

    _log(job_id, f"[{idx}/{total}] Processing: {obj_id} - {obj_text[:70]}...")

    conn = base.db_connect()
    conn.execute(
        """
        UPDATE objectives
        SET status=?, started_at_unix=?, error_text=NULL
        WHERE job_id=? AND objective_id=?
        """,
        ("running", int(time.time()), job_id, obj_id),
    )
    conn.commit()
    conn.close()

    start = time.time()
    ik: Optional[Dict[str, Any]] = None
    err: Optional[str] = None

    try:
        ik = base.generate_for_objective_strict(
            base_url=ollama_url,
            worker_model=worker_model,
            manager_model=manager_model,
            objective_id=obj_id,
            objective_text=obj_text,
            max_rounds=max_rounds,
            num_ctx=num_ctx,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        base.validate_ikizamini(ik)
        elapsed = time.time() - start
        _log(job_id, f"  ✓ Success ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start
        err = str(e)
        _log(job_id, f"  ✗ FAILED ({elapsed:.1f}s): {obj_id}: {err}")
        with failures_lock:
            failures.append(f"{obj_id}: {err}")

    txt_path, json_path = base.write_output_files(job_id, obj_id, obj_text, path, ik, err)

    conn = base.db_connect()
    conn.execute(
        """
        UPDATE objectives
        SET status=?, finished_at_unix=?, error_text=?, output_txt_path=?, output_json_path=?
        WHERE job_id=? AND objective_id=?
        """,
        (
            "done" if ik is not None else "failed",
            int(time.time()),
            err,
            txt_path,
            json_path or None,
            job_id,
            obj_id,
        ),
    )
    conn.commit()
    conn.close()

    with failures_lock:
        _update_job_progress(job_id, failures)


def run_job(job_id: str, resume: bool = False) -> None:
    """Parallel objective processing version of ikizamini_local.run_job()."""
    try:
        with base.RUNNER_LOCK:
            conn = base.db_connect()
            cur = conn.cursor()
            cur.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
            job = cur.fetchone()
            conn.close()

            if not job:
                return

            cfg = json.loads(job["config_json"])
            ollama_url = cfg["ollama_url"]
            worker_model = cfg["worker_model"]
            manager_model = cfg["manager_model"]
            max_rounds = int(cfg["max_rounds"])
            num_ctx = int(cfg["num_ctx"])
            limit = int(cfg.get("limit", 0))
            timeout_s = int(cfg.get("timeout_s", base.DEFAULT_TIMEOUT_S))
            max_retries = int(cfg.get("max_retries", base.DEFAULT_MAX_RETRIES))

            # Diagnostics: show whether Ollama thinks it is using CPU/GPU for the loaded model(s)
            try:
                ps_out = subprocess.check_output(["ollama", "ps"], text=True, stderr=subprocess.STDOUT)
                _log(job_id, "Ollama ps (diagnostic):")
                for ln in ps_out.strip().splitlines()[:12]:
                    _log(job_id, ln)
            except Exception:
                _log(job_id, "Ollama ps (diagnostic): unavailable")

            input_path = job["input_path"]
            if not input_path or not os.path.exists(input_path):
                _log(job_id, "ERROR: input file missing; cannot start.")
                conn = base.db_connect()
                conn.execute(
                    "UPDATE jobs SET status=?, finished_at_unix=? WHERE job_id=?",
                    ("error", int(time.time()), job_id),
                )
                conn.commit()
                conn.close()
                return

            conn = base.db_connect()
            conn.execute(
                "UPDATE jobs SET status=?, started_at_unix=?, heartbeat_unix=? WHERE job_id=?",
                ("running", job["started_at_unix"] or int(time.time()), int(time.time()), job_id),
            )
            conn.commit()
            conn.close()

            _log(job_id, "============================================================")
            _log(job_id, "BACKGROUND GENERATION STARTED (PARALLEL)")
            _log(job_id, f"Ollama URL: {ollama_url}")
            _log(job_id, f"Worker model: {worker_model}")
            _log(job_id, f"Manager model: {manager_model}")
            _log(job_id, f"Max rounds: {max_rounds}  num_ctx: {num_ctx}")
            _log(job_id, f"Timeout: {timeout_s}s  Retries: {max_retries}")
            _log(job_id, f"Objective concurrency: {PARALLEL_WORKERS}")
            _log(job_id, "============================================================")

            with open(input_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

        roots = base.parse_objectives(content)
        objectives = base.collect_learning_objectives_with_paths(roots)
        if limit and limit > 0:
            objectives = objectives[:limit]

        # Ensure objective rows exist
        conn = base.db_connect()
        cur = conn.cursor()
        for obj in objectives:
            cur.execute(
                """
                INSERT OR IGNORE INTO objectives
                (job_id, objective_id, objective_text, path_json, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    obj["objective_id"],
                    obj["objective_text"],
                    json.dumps(obj["path"], ensure_ascii=False),
                    "pending",
                ),
            )
        conn.commit()
        conn.close()

        conn = base.db_connect()
        conn.execute("UPDATE jobs SET requested=? WHERE job_id=?", (len(objectives), job_id))
        conn.commit()
        conn.close()

        failures: List[str] = json.loads(job["failures_json"] or "[]")
        failures_lock = threading.Lock()

        # Filter objectives if resuming
        to_run: List[Dict[str, Any]] = []
        if resume:
            conn = base.db_connect()
            cur = conn.cursor()
            for obj in objectives:
                cur.execute(
                    "SELECT status FROM objectives WHERE job_id=? AND objective_id=?",
                    (job_id, obj["objective_id"]),
                )
                row = cur.fetchone()
                if row and row["status"] in ("done", "failed"):
                    continue
                to_run.append(obj)
            conn.close()
        else:
            to_run = list(objectives)

        # Heartbeat thread
        stop_hb = threading.Event()

        def _hb_loop():
            while not stop_hb.is_set():
                base.update_job_heartbeat(job_id)
                stop_hb.wait(base.HEARTBEAT_EVERY_SEC)

        hb_thread = threading.Thread(target=_hb_loop, daemon=True)
        hb_thread.start()

        # Parallel objective execution (supports termination: stop scheduling new work)
        total = len(to_run)
        cancelling = False
        with ThreadPoolExecutor(max_workers=max(1, PARALLEL_WORKERS)) as ex:
            futs: Dict[Any, Tuple[int, Dict[str, Any]]] = {}
            it = iter(list(enumerate(to_run, start=1)))

            def submit_next() -> bool:
                nonlocal cancelling
                if cancelling or _cancel_requested(job_id):
                    cancelling = True
                    return False
                try:
                    i, obj = next(it)
                except StopIteration:
                    return False
                fut = ex.submit(
                    _process_objective,
                    job_id,
                    obj,
                    i,
                    total,
                    ollama_url,
                    worker_model,
                    manager_model,
                    max_rounds,
                    num_ctx,
                    timeout_s,
                    max_retries,
                    failures,
                    failures_lock,
                )
                futs[fut] = (i, obj)
                return True

            # Prefill
            for _ in range(max(1, PARALLEL_WORKERS)):
                if not submit_next():
                    break

            # Main loop: wait for completions and keep queue filled while not cancelling
            while futs:
                done, _ = wait(set(futs.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    futs.pop(fut, None)
                    try:
                        fut.result()
                    except Exception as e:
                        _log(job_id, f"INTERNAL ERROR in worker thread: {e}")
                        with failures_lock:
                            failures.append(f"internal: {e}")

                if _cancel_requested(job_id) and not cancelling:
                    cancelling = True
                    _log(job_id, "CANCEL REQUESTED: stopping scheduling new objectives (parallel).")
                    conn = base.db_connect()
                    conn.execute(
                        "UPDATE jobs SET status=?, heartbeat_unix=? WHERE job_id=?",
                        ("cancelling", int(time.time()), job_id),
                    )
                    conn.commit()
                    conn.close()
                    base.mark_remaining_objectives_cancelled(job_id)  # type: ignore

                # Keep filling queue if not cancelling
                while len(futs) < max(1, PARALLEL_WORKERS):
                    if not submit_next():
                        break

        stop_hb.set()

        final_status = "cancelled" if _cancel_requested(job_id) else "done"
        if final_status == "cancelled":
            base.mark_remaining_objectives_cancelled(job_id)  # type: ignore
        conn = base.db_connect()
        conn.execute(
            "UPDATE jobs SET status=?, finished_at_unix=?, heartbeat_unix=? WHERE job_id=?",
            (final_status, int(time.time()), int(time.time()), job_id),
        )
        conn.commit()
        conn.close()

        _log(job_id, "============================================================")
        _log(job_id, f"JOB COMPLETED: {job_id}  (outputs persisted to disk + DB)")
        _log(job_id, "============================================================")
    except Exception as e:
        # Ensure UI doesn't look stuck forever if the runner crashes.
        try:
            _log(job_id, f"FATAL: job runner crashed: {e}")
            conn = base.db_connect()
            conn.execute(
                "UPDATE jobs SET status=?, finished_at_unix=?, heartbeat_unix=? WHERE job_id=?",
                ("error", int(time.time()), int(time.time()), job_id),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass


# Override the base runner used by the existing UI routes.
base.run_job = run_job  # type: ignore

# Re-export app and startup so running this file behaves like ikizamini_local.py
app = base.app
startup = base.startup


if __name__ == "__main__":
    os.environ.setdefault("IKIZAMINI_UI_TITLE", "IKIZAMINI Parallel Processor")
    startup()
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port_env = os.environ.get("RUNPOD_PORT") or os.environ.get("FLASK_PORT")
    if port_env:
        port = int(port_env)
    else:
        port = base._pick_free_port(host, [8000, 8001, 5000, 5001, 8080, 8081])  # type: ignore
    base.safe_print(f"[{base._now_ts()}] Starting IKIZAMINI UI (PARALLEL) on {host}:{port} ...")  # type: ignore
    app.run(host=host, port=port, debug=False, threaded=True)

