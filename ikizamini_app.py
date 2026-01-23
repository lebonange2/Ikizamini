#!/usr/bin/env python3
"""
ikizamini_app.py

Fixes included:
- Robust whitespace normalization (handles NBSP and other Unicode spaces).
- Only generates for true learning objectives (id depth >= 4, e.g. 1.1.1.1).
- Prevents sub-topic nodes (e.g. 1.1.1) from being selected even if parsing makes them appear as leaves.
- Strict repair loop: worker fixes issues, manager approves, final output always schema-valid.
- Master file preserves input heading structure; optional per-objective files.

Requirements:
  pip install openai jsonschema

Usage:
    python3 ikizamini_app.py --input Uru.txt --output ikizamini.txt --output-dir out_ikizamini

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import validate as jsonschema_validate, ValidationError
from openai import OpenAI


# ----------------------------
# Objective tree parsing
# ----------------------------

@dataclass
class ObjNode:
    obj_id: str
    title: str
    indent: int
    children: List["ObjNode"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# Matches: "1.1.1.1 Some title" (tolerates weird spacing after id)
OBJ_LINE_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.*\S)\s*$")

# Unicode spaces commonly seen in copied outlines (NBSP etc.)
UNICODE_SPACES = [
    "\u00A0",  # NO-BREAK SPACE
    "\u2007",  # FIGURE SPACE
    "\u202F",  # NARROW NO-BREAK SPACE
    "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005",
    "\u2006", "\u2008", "\u2009", "\u200A",
    "\u205F",  # MEDIUM MATHEMATICAL SPACE
    "\u3000",  # IDEOGRAPHIC SPACE
]


def normalize_outline_text(text: str) -> str:
    """
    Make the outline parseable:
    - Convert unicode spaces to normal spaces
    - Convert tabs to 4 spaces
    - Normalize line endings
    """
    for sp in UNICODE_SPACES:
        text = text.replace(sp, " ")
    text = text.replace("\t", "    ")
    # Normalize weird multiple spaces (keep indentation intact by only trimming end)
    text = "\n".join([ln.rstrip() for ln in text.splitlines()])
    return text


def parse_objectives(text: str) -> List[ObjNode]:
    """
    Parse a numbered + indented outline into ObjNodes using indentation as hierarchy.
    Whitespace is normalized before parsing to reduce indentation glitches.
    """
    text = normalize_outline_text(text)
    lines = text.splitlines()

    roots: List[ObjNode] = []
    stack: List[ObjNode] = []

    for raw in lines:
        if not raw.strip():
            continue

        m = OBJ_LINE_RE.match(raw)
        if not m:
            continue

        obj_id = m.group(1).strip()
        title = m.group(2).strip()

        # Indent count based on leading spaces after normalization
        indent = len(raw) - len(raw.lstrip(" "))

        node = ObjNode(obj_id=obj_id, title=title, indent=indent)

        # Parent selection by indentation
        while stack and indent <= stack[-1].indent:
            stack.pop()

        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)

        stack.append(node)

    return roots


def id_depth(obj_id: str) -> int:
    # "1.1.1.10" -> 4 segments
    return len([p for p in obj_id.split(".") if p.strip() != ""])


def is_learning_objective_id(obj_id: str) -> bool:
    """
    In your structure, learning objectives are at least 4 segments:
    1.1.1.1, 1.1.1.2, ...
    Sub-topics like 1.1.1 should be excluded.
    """
    return id_depth(obj_id) >= 4


def collect_learning_objectives_with_paths(roots: List[ObjNode]) -> List[Dict[str, Any]]:
    """
    Collect ONLY learning objectives (id depth >= 4).
    Do not rely on is_leaf(), because indentation glitches can misclassify nodes.
    """
    out: List[Dict[str, Any]] = []

    def dfs(n: ObjNode, path: List[Tuple[str, str]]):
        new_path = path + [(n.obj_id, n.title)]

        if is_learning_objective_id(n.obj_id):
            out.append(
                {
                    "objective_id": n.obj_id,
                    "objective_text": n.title,
                    "path": new_path,
                }
            )
        else:
            for c in n.children:
                dfs(c, new_path)

    for r in roots:
        dfs(r, [])
    return out


# ----------------------------
# OpenAI structured output calls
# ----------------------------

def backoff_sleep(attempt: int) -> None:
    base = min(2 ** attempt, 16)
    jitter = random.uniform(0.0, 0.5)
    time.sleep(base + jitter)


def call_openai_structured(
    client: OpenAI,
    model: str,
    schema_name: str,
    schema: Dict[str, Any],
    system_instructions: str,
    user_prompt: str,
    max_retries: int = 5,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_prompt},
                ],
                store=False,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
            raw = resp.output_text.strip()
            return json.loads(raw)
        except Exception as e:
            last_err = e
            backoff_sleep(attempt)
    raise RuntimeError(f"OpenAI call failed after retries. Last error: {last_err}")


# ----------------------------
# Schemas
# ----------------------------

IKIZAMINI_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "objective_id": {"type": "string"},
        "objective_text": {"type": "string"},
        "difficulty_profile": {"type": "string"},
        "questions": {
            "type": "array",
            "minItems": 7,
            "maxItems": 7,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "stem": {"type": "string"},
                    "choices": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "A": {"type": "string"},
                            "B": {"type": "string"},
                            "C": {"type": "string"},
                            "D": {"type": "string"},
                        },
                        "required": ["A", "B", "C", "D"],
                    },
                    "correct_choice": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    "solution": {"type": "string"},
                    "explanation": {"type": "string"},
                    "common_mistake_notes": {"type": "string"},
                },
                "required": [
                    "id",
                    "stem",
                    "choices",
                    "correct_choice",
                    "solution",
                    "explanation",
                    "common_mistake_notes",
                ],
            },
        },
    },
    "required": ["objective_id", "objective_text", "difficulty_profile", "questions"],
}

MANAGER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["pass", "fail"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "revised_questions": {
            "type": "array",
            "minItems": 7,
            "maxItems": 7,
            "items": IKIZAMINI_SCHEMA["properties"]["questions"]["items"],
        },
        "score": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "relevance": {"type": "integer", "minimum": 0, "maximum": 5},
                "distinctness": {"type": "integer", "minimum": 0, "maximum": 5},
                "unique_solution": {"type": "integer", "minimum": 0, "maximum": 5},
                "explanations": {"type": "integer", "minimum": 0, "maximum": 5},
            },
            "required": ["relevance", "distinctness", "unique_solution", "explanations"],
        },
    },
    "required": ["status", "issues", "revised_questions", "score"],
}


def validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
    jsonschema_validate(instance=data, schema=schema)


# ----------------------------
# Agent prompts
# ----------------------------

WORKER_SYSTEM = """You are workerAgent.
You create multiple-choice exam questions for a specific learning objective.

Hard requirements:
- Produce EXACTLY 7 questions.
- Each question must be DISTINCT in concept and surface form.
- Exactly 4 answer choices labeled A, B, C, D.
- Exactly ONE correct answer (no ambiguity; no multiple correct).
- Provide a worked solution and a clear explanation.
- Include common_mistake_notes describing one likely mistake and why it fails.
- Keep problems aligned to the objective; do not drift.
- Provide difficulty_profile as a short string (e.g., "easy→hard mixed" or "mixed").

Return ONLY JSON that matches the given schema.
"""

WORKER_REPAIR_SYSTEM = """You are workerAgent in REPAIR mode.
Fix the provided set so it:
- matches the JSON schema exactly (no extra keys, no missing keys),
- has correct math and solutions,
- has exactly ONE correct choice per question,
- has 7 distinct questions aligned to the objective,
- includes solution + explanation + common mistake notes for each question.

Return ONLY JSON that matches the given schema.
"""

MANAGER_SYSTEM = """You are managerAgent.
Review the 7 exam questions for the learning objective.

Enforce:
1) Relevance
2) Distinctness
3) Unique solution
4) Explanations correctness/clarity

Output rules:
- Always return revised_questions (full set of 7).
- If pass: revised_questions should match input (minor normalization ok).
- If fail: revised_questions must be corrected to satisfy criteria.
Return ONLY JSON matching the schema.
"""


def worker_generate(client: OpenAI, model: str, objective_id: str, objective_text: str) -> Dict[str, Any]:
    prompt = f"""Learning objective:
- id: {objective_id}
- text: {objective_text}

Generate 7 exam questions that directly assess this objective."""
    return call_openai_structured(
        client=client,
        model=model,
        schema_name="ikizamini_set",
        schema=IKIZAMINI_SCHEMA,
        system_instructions=WORKER_SYSTEM,
        user_prompt=prompt,
    )


def worker_repair(
    client: OpenAI,
    model: str,
    objective_id: str,
    objective_text: str,
    candidate: Dict[str, Any],
    issues: List[str],
) -> Dict[str, Any]:
    prompt = f"""Learning objective:
- id: {objective_id}
- text: {objective_text}

Issues to fix:
{json.dumps(issues, ensure_ascii=False, indent=2)}

Current JSON:
{json.dumps(candidate, ensure_ascii=False)}

Return a fully corrected JSON set."""
    return call_openai_structured(
        client=client,
        model=model,
        schema_name="ikizamini_set",
        schema=IKIZAMINI_SCHEMA,
        system_instructions=WORKER_REPAIR_SYSTEM,
        user_prompt=prompt,
    )


def manager_review(
    client: OpenAI,
    model: str,
    objective_id: str,
    objective_text: str,
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = f"""Learning objective:
- id: {objective_id}
- text: {objective_text}

Candidate JSON:
{json.dumps(candidate, ensure_ascii=False)}

Review and return status, issues, score, revised_questions."""
    return call_openai_structured(
        client=client,
        model=model,
        schema_name="manager_review",
        schema=MANAGER_SCHEMA,
        system_instructions=MANAGER_SYSTEM,
        user_prompt=prompt,
    )


def local_issues(candidate: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    qs = candidate.get("questions", [])
    if len(qs) != 7:
        issues.append(f"Expected exactly 7 questions; got {len(qs)}.")
    stems = [q.get("stem", "").strip() for q in qs if isinstance(q, dict)]
    if len(stems) != len(set(stems)):
        issues.append("Duplicate stems detected (exact match).")
    for i, q in enumerate(qs, start=1):
        if not isinstance(q, dict):
            continue
        ch = q.get("choices", {})
        if not isinstance(ch, dict):
            continue
        vals = [str(ch.get(k, "")).strip() for k in ["A", "B", "C", "D"]]
        if len(vals) == 4 and len(set(vals)) != 4:
            issues.append(f"Q{i}: duplicate choice text detected.")
    return issues


def generate_for_objective_strict(
    client: OpenAI,
    model: str,
    objective_id: str,
    objective_text: str,
    max_rounds: int,
) -> Dict[str, Any]:
    candidate = worker_generate(client, model, objective_id, objective_text)

    for _ in range(max_rounds):
        issues: List[str] = []

        # Schema validation
        try:
            validate_against_schema(candidate, IKIZAMINI_SCHEMA)
        except ValidationError as ve:
            issues.append(f"Schema validation error: {ve.message}")

        # Local checks
        issues.extend(local_issues(candidate))

        if issues:
            candidate = worker_repair(client, model, objective_id, objective_text, candidate, issues)
            continue

        review = manager_review(client, model, objective_id, objective_text, candidate)
        validate_against_schema(review, MANAGER_SCHEMA)

        # Apply manager revised questions
        candidate = {
            "objective_id": objective_id,
            "objective_text": objective_text,
            "difficulty_profile": candidate.get("difficulty_profile", "mixed") or "mixed",
            "questions": review["revised_questions"],
        }

        # Revalidate
        try:
            validate_against_schema(candidate, IKIZAMINI_SCHEMA)
        except ValidationError as ve:
            candidate = worker_repair(
                client, model, objective_id, objective_text, candidate,
                [f"Schema validation error after manager edits: {ve.message}"]
            )
            continue

        if review["status"] == "pass":
            return candidate

        # Manager failed; repair with manager issues
        mgr_issues = review.get("issues", []) or ["Manager marked fail without detailed issues."]
        candidate = worker_repair(client, model, objective_id, objective_text, candidate, mgr_issues)

    raise RuntimeError(f"Could not reach manager approval after {max_rounds} rounds for objective {objective_id}.")


# ----------------------------
# Output formatting
# ----------------------------

def safe_filename(s: str, max_len: int = 90) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\-. ]+", "", s)
    s = s.replace(" ", "_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "objective"


def unique_objective_filename(objective_id: str, objective_text: str) -> str:
    base = f"{objective_id}|{objective_text}".encode("utf-8")
    suffix = hashlib.sha256(base).hexdigest()[:8]
    return f"{objective_id}_{safe_filename(objective_text)}_{suffix}.txt"


def format_questions_block(ikizamini_set: Dict[str, Any]) -> str:
    lines: List[str] = []
    for idx, q in enumerate(ikizamini_set["questions"], start=1):
        lines.append(f"\nProblem {idx} ({q['id']}):")
        lines.append(q["stem"].strip())
        ch = q["choices"]
        lines.append(f"A) {ch['A']}")
        lines.append(f"B) {ch['B']}")
        lines.append(f"C) {ch['C']}")
        lines.append(f"D) {ch['D']}")
        lines.append(f"Correct Answer: {q['correct_choice']}")
        lines.append("\nSolution:")
        lines.append(q["solution"].strip())
        lines.append("\nExplanation:")
        lines.append(q["explanation"].strip())
        lines.append("\nCommon mistake note:")
        lines.append(q["common_mistake_notes"].strip())
        lines.append("-" * 78)
    return "\n".join(lines).strip() + "\n"


def write_per_objective_file(
    output_dir: str,
    path: List[Tuple[str, str]],
    objective_id: str,
    objective_text: str,
    ikizamini_set: Optional[Dict[str, Any]] = None,
    error_text: Optional[str] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, unique_objective_filename(objective_id, objective_text))

    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(f"LEARNING OBJECTIVE: {objective_id} {objective_text}")
    lines.append("=" * 78)

    if len(path) > 1:
        breadcrumb = " > ".join([f"{pid} {ptitle}" for pid, ptitle in path[:-1]])
        lines.append(f"Context: {breadcrumb}")
        lines.append("-" * 78)

    if ikizamini_set is not None:
        lines.append(f"Difficulty profile: {ikizamini_set.get('difficulty_profile', 'mixed')}")
        lines.append("-" * 78)
        lines.append(format_questions_block(ikizamini_set))
    else:
        lines.append("STATUS: FAILED TO GENERATE QUESTIONS FOR THIS OBJECTIVE")
        lines.append("-" * 78)
        lines.append((error_text or "Unknown error.").strip())
        lines.append("\nSuggested action: increase --max-rounds or check API connectivity.\n")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    return file_path


def write_master_from_tree(
    roots: List[ObjNode],
    generated_by_id: Dict[str, Dict[str, Any]],
    output_path: str,
    meta: Dict[str, Any],
) -> None:
    def heading_line(node: ObjNode) -> str:
        return f"{node.obj_id} {node.title}"

    def render(node: ObjNode, level: int) -> List[str]:
        indent = "   " * max(level - 1, 0)
        out: List[str] = [f"{indent}{heading_line(node)}"]

        # Only insert questions for true learning objectives
        if is_learning_objective_id(node.obj_id):
            ik = generated_by_id.get(node.obj_id)
            if ik:
                out.append(f"{indent}{'-' * 78}")
                block = format_questions_block(ik).replace("\n", "\n" + indent)
                out.append(indent + block.rstrip())
            else:
                out.append(f"{indent}(No generated output for this objective.)")
        else:
            for c in node.children:
                out.extend(render(c, level + 1))
        return out

    if not output_path.lower().endswith(".txt"):
        output_path += ".txt"

    lines: List[str] = []
    lines.append("IKIZAMINI MASTER OUTPUT")
    lines.append(f"Source file: {meta.get('source_file', '')}")
    lines.append(f"Model: {meta.get('model', '')}")
    lines.append(f"Generated at (unix): {meta.get('generated_at_unix', '')}")
    lines.append(f"Objectives completed: {meta.get('completed', 0)} / {meta.get('requested', 0)}")
    lines.append("\n" + "#" * 78 + "\n")

    for r in roots:
        lines.extend(render(r, 1))
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate exam questions for learning objectives only (1.1.1.1+).")
    ap.add_argument("--input", required=True, help="Path to objectives .txt file")
    ap.add_argument("--output", required=True, help="Path to master .txt output file (contains ALL objectives)")
    ap.add_argument("--output-dir", default="", help="Optional: directory for per-objective .txt files")
    ap.add_argument("--index-file", default="index.txt", help="Index filename inside --output-dir (only if output-dir is used)")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI model (default: gpt-4o)")
    ap.add_argument("--max-rounds", type=int, default=5, help="Max repair/review rounds per objective (default: 5)")
    ap.add_argument("--limit", type=int, default=0, help="Only process first N learning objectives (0 = all)")
    args = ap.parse_args()

    client = OpenAI()

    with open(args.input, "r", encoding="utf-8") as f:
        raw = f.read()

    roots = parse_objectives(raw)

    # IMPORTANT: only learning objectives (depth >= 4)
    objectives = collect_learning_objectives_with_paths(roots)

    if args.limit and args.limit > 0:
        objectives = objectives[: args.limit]

    generated_by_id: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    per_written: List[str] = []

    for idx, leaf in enumerate(objectives, start=1):
        obj_id = leaf["objective_id"]
        obj_text = leaf["objective_text"]
        path = leaf["path"]

        print(f"[{idx}/{len(objectives)}] Generating exam questions for {obj_id} {obj_text}")
        try:
            ik = generate_for_objective_strict(
                client=client,
                model=args.model,
                objective_id=obj_id,
                objective_text=obj_text,
                max_rounds=args.max_rounds,
            )
            validate_against_schema(ik, IKIZAMINI_SCHEMA)

            generated_by_id[obj_id] = ik

            if args.output_dir:
                per_written.append(write_per_objective_file(args.output_dir, path, obj_id, obj_text, ikizamini_set=ik))

        except Exception as e:
            failures.append(f"{obj_id}: {e}")
            print(f"Failed for objective {obj_id}: {e}", file=sys.stderr)

            if args.output_dir:
                per_written.append(write_per_objective_file(args.output_dir, path, obj_id, obj_text, ikizamini_set=None, error_text=str(e)))

    meta = {
        "source_file": os.path.basename(args.input),
        "model": args.model,
        "generated_at_unix": int(time.time()),
        "requested": len(objectives),
        "completed": len(generated_by_id),
    }
    write_master_from_tree(roots, generated_by_id, args.output, meta)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        index_path = os.path.join(args.output_dir, args.index_file)
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("IKIZAMINI PER-OBJECTIVE INDEX\n")
            f.write(f"Source file: {meta['source_file']}\n")
            f.write(f"Model: {meta['model']}\n")
            f.write(f"Generated at (unix): {meta['generated_at_unix']}\n")
            f.write(f"Learning objectives requested: {meta['requested']}\n")
            f.write(f"Learning objectives completed: {meta['completed']}\n\n")

            f.write("FILES:\n")
            for p in per_written:
                f.write(f"- {os.path.basename(p)}\n")

            if failures:
                f.write("\nFAILURES:\n")
                for x in failures:
                    f.write(f"- {x}\n")

        print(f"Per-objective index written to {index_path}")

    out_path = args.output if args.output.lower().endswith(".txt") else args.output + ".txt"
    print(f"Done. Master output written to {out_path}")
    if failures:
        print(f"Completed with {len(failures)} failures (see console or per-objective index if enabled).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
