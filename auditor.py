#!/usr/bin/env python3
"""
auditor.py

auditor: Multi-agent auditor for Q/A .txt files under output/.

Changes per request:
- Renamed AUDITRIX -> auditor everywhere (labels, comments, UI strings).
- Corrected files are saved same name as original (no suffix/prefix changes),
  preserving identical relative path under audited_output/.

Outputs:
- Corrected files: audited_output/<same_subdirs>/<same_filename>.txt
- Per-file audit report: audit_reports/<same_subdirs>/<stem>_audit.json
- Master summary: audit_reports/MASTER_SUMMARY.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI  # reads OPENAI_API_KEY from environment
from jsonschema import validate as jsonschema_validate
from jsonschema import ValidationError

MODEL_NAME = "gpt-5.2"

AUDIT_ITEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question_id": {"type": "string"},
        "question_excerpt": {"type": "string"},
        "provided_answer": {"type": "string"},
        "verdict": {"type": "string", "enum": ["verified", "corrected", "unverified", "needs_external_source"]},
        "correct_answer": {"type": ["string", "null"]},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "tags": {
            "type": "array",
            "items": {"type": "string", "enum": ["math", "science", "language", "other"]},
            "minItems": 1,
        },
    },
    "required": [
        "question_id",
        "question_excerpt",
        "provided_answer",
        "verdict",
        "correct_answer",
        "reasoning",
        "confidence",
        "tags",
    ],
}

PER_FILE_REPORT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "input_file": {"type": "string"},
        "output_file": {"type": "string"},
        "items": {"type": "array", "items": AUDIT_ITEM_SCHEMA},
        "stats": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "total_items": {"type": "integer", "minimum": 0},
                "verified": {"type": "integer", "minimum": 0},
                "corrected": {"type": "integer", "minimum": 0},
                "unverified": {"type": "integer", "minimum": 0},
                "needs_external_source": {"type": "integer", "minimum": 0},
            },
            "required": ["total_items", "verified", "corrected", "unverified", "needs_external_source"],
        },
    },
    "required": ["input_file", "output_file", "items", "stats"],
}

PARSER_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question_id": {"type": "string"},
                    "format_type": {"type": "string"},
                    "question_line_start": {"type": "integer", "minimum": 1},
                    "question_line_end": {"type": "integer", "minimum": 1},
                    "answer_line_start": {"type": ["integer", "null"], "minimum": 1},
                    "answer_line_end": {"type": ["integer", "null"], "minimum": 1},
                    "question_text": {"type": "string"},
                    "provided_answer_text": {"type": "string"},
                    "parse_uncertain": {"type": "boolean"},
                },
                "required": [
                    "question_id",
                    "format_type",
                    "question_line_start",
                    "question_line_end",
                    "answer_line_start",
                    "answer_line_end",
                    "question_text",
                    "provided_answer_text",
                    "parse_uncertain",
                ],
            },
        }
    },
    "required": ["items"],
}

AUDITOR_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["verified", "corrected", "unverified", "needs_external_source"]},
        "correct_answer": {"type": ["string", "null"]},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "tags": {
            "type": "array",
            "items": {"type": "string", "enum": ["math", "science", "language", "other"]},
            "minItems": 1,
        },
    },
    "required": ["verdict", "correct_answer", "reasoning", "confidence", "tags"],
}

CROSSCHECK_OUTPUT_SCHEMA = AUDITOR_OUTPUT_SCHEMA

QUALITY_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["pass", "fail"]},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "issues"],
}

MANAGER_SYSTEM = """You are auditor Manager Agent (Orchestrator).
You do not answer questions directly. You enforce process discipline:
- No hallucination.
- Preserve question wording intact.
- Ensure answer is single and unambiguous unless question explicitly allows multiple.
- Ensure math/logic corrections include key steps in audit reasoning.
You will not output anything unless asked for JSON in a tool step."""

PARSER_SYSTEM = """You are auditor Parser Agent.
Task: Extract Question → Provided Answer pairs from a raw .txt file with LINE NUMBERS.

You MUST:
- Identify each question block and its corresponding provided answer block.
- Support formats: "Q:"/"A:", numbered questions with "Answer:", multiple-choice blocks, inline answers.
- Return JSON ONLY matching the given schema.
- Provide accurate line number spans:
  - question_line_start/end: inclusive line numbers for the question block
  - answer_line_start/end: inclusive span for the provided answer block; null if not found
- question_text must be exactly the text in question span (preserve punctuation and wording)
- provided_answer_text must be exactly the text in answer span (or empty string if missing)
- If uncertain, set parse_uncertain=true but still provide best-effort spans.

Do NOT invent content. Do NOT rewrite the question. Do NOT add commentary. Return JSON only.
"""

AUDITOR_COMMON_RULES = """Hard rules:
- No hallucination: if verifying requires missing external reference/data, set verdict="needs_external_source".
- If the provided answer is correct and complete, verdict="verified" and correct_answer=null.
- If wrong/incomplete, verdict="corrected" and correct_answer must be the corrected answer.
- If you cannot verify with available info but no external reference is strictly required, verdict="unverified".
- Show key steps in reasoning for math/logic (concise, but include the justification).
- Be precise: avoid ambiguous answers. If question is MCQ, include option letter AND final value when possible (e.g., "(C) 12").
- Return JSON only matching schema.
"""

MATH_AUDITOR_SYSTEM = f"""You are auditor Math Auditor.
You verify algebra, calculus, geometry, arithmetic, logic puzzles.
{AUDITOR_COMMON_RULES}
"""

SCIENCE_AUDITOR_SYSTEM = f"""You are auditor Science/Engineering Auditor.
You verify physics, circuits, engineering fundamentals, units, dimensional reasoning.
{AUDITOR_COMMON_RULES}
"""

LANGUAGE_AUDITOR_SYSTEM = f"""You are auditor General Knowledge/Language Auditor.
You verify definitions, conceptual explanations, reading comprehension, grammar.
{AUDITOR_COMMON_RULES}
"""

CROSSCHECK_SYSTEM = f"""You are auditor Cross-Checker Agent.
Independently re-verify the item and any proposed correction.
- If you agree the correction is defensible, keep it.
- If you find an error, provide the most defensible answer with reasoning.
- If still uncertain, mark verdict="unverified" with rationale.
- Follow no hallucination and show key steps when needed.
Return JSON only matching schema.
"""

QUALITY_SYSTEM = """You are auditor Quality & Consistency Agent.
Check that:
- Corrected file only changes answer portions using required labels:
  Answer (Verified), Answer (Corrected), Answer (Unverified), Answer (Needs External Source)
- Question text is unchanged.
- No extra commentary is inserted beyond the answer label line(s).
Return JSON only:
{ "status": "pass|fail", "issues": ["..."] }
"""


def backoff_sleep(attempt: int) -> None:
    base = min(2 ** attempt, 16)
    jitter = random.uniform(0.0, 0.5)
    time.sleep(base + jitter)


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def extract_json_value(text: str) -> Any:
    m = JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            val, _ = decoder.raw_decode(text[i:])
            return val
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON found in model output.")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, s: str) -> None:
    safe_mkdir(path.parent)
    path.write_text(s, encoding="utf-8")


def normalize_excerpt(s: str, n: int = 160) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s[:n]


def detect_tag(question_text: str) -> str:
    qt = question_text.lower()
    if any(tok in qt for tok in ["integral", "derivative", "solve", "simplify", "factor", "equation", "sqrt", "∛", "π", "sin", "cos", "tan", "log", "^", "x^", "y="]):
        return "math"
    if any(tok in qt for tok in ["voltage", "current", "resistor", "ohm", "circuit", "newton", "force", "energy", "joule", "watt", "amp", "capacitor", "inductor", "ttl", "transistor"]):
        return "science"
    if any(tok in qt for tok in ["define", "definition", "explain", "describe", "grammar", "meaning", "summarize"]):
        return "language"
    return "other"


def error_category_from_reasoning(reasoning: str, verdict: str) -> str:
    r = reasoning.lower()
    if verdict in ("unverified", "needs_external_source"):
        return "insufficient_information"
    if any(k in r for k in ["arithmetic", "calculation", "compute", "miscalculate"]):
        return "arithmetic_error"
    if any(k in r for k in ["algebra", "factor", "simplify", "manipulation"]):
        return "algebraic_error"
    if any(k in r for k in ["unit", "dimension", "convert"]):
        return "unit_mismatch"
    if any(k in r for k in ["concept", "definition", "misunderstand"]):
        return "conceptual_misunderstanding"
    if any(k in r for k in ["ambiguous", "multiple correct", "more than one"]):
        return "ambiguity_multiple_solutions"
    if any(k in r for k in ["incomplete", "missing step", "not fully"]):
        return "incomplete_answer"
    return "other"


@dataclass
class AgentCallResult:
    data: Any
    raw_text: str


class OpenAIOrchestrator:
    def __init__(self) -> None:
        self.client = OpenAI()

    def call_json_agent(
        self,
        system: str,
        user: str,
        schema: Dict[str, Any],
        max_retries: int = 4,
        temperature: float = 0.2,
    ) -> AgentCallResult:
        last_err: Optional[Exception] = None
        raw = ""
        for attempt in range(max_retries):
            try:
                resp = self.client.responses.create(
                    model=MODEL_NAME,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                )
                raw = (resp.output_text or "").strip()
                val = extract_json_value(raw)
                jsonschema_validate(instance=val, schema=schema)
                return AgentCallResult(data=val, raw_text=raw)
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    user = (
                        "Your previous output did not validate as strict JSON for the required schema.\n"
                        "Return ONLY corrected JSON that validates. No markdown, no commentary.\n\n"
                        f"Validation error: {str(e)}\n\n"
                        "Original task:\n"
                        f"{user}"
                    )
                    backoff_sleep(attempt)
        raise RuntimeError(f"OpenAI call failed after retries. Last error: {last_err}. Raw preview: {raw[:300]}")


def parser_extract_items(orch: OpenAIOrchestrator, file_text: str) -> List[Dict[str, Any]]:
    numbered = []
    for idx, line in enumerate(file_text.splitlines(), start=1):
        numbered.append(f"{idx:05d}: {line}")
    numbered_text = "\n".join(numbered)

    user = (
        "Parse this text into Question → Provided Answer pairs.\n"
        "The text includes line numbers prefixed as '00001: '. Use them.\n\n"
        "Return JSON with structure:\n"
        '{ "items": [ { question_id, format_type, question_line_start, question_line_end, '
        "answer_line_start, answer_line_end, question_text, provided_answer_text, parse_uncertain } ] }\n\n"
        "INPUT:\n"
        f"{numbered_text}"
    )
    res = orch.call_json_agent(PARSER_SYSTEM, user, PARSER_OUTPUT_SCHEMA, temperature=0.1)
    items = res.data["items"]
    cleaned = []
    for it in items:
        if it["question_line_end"] < it["question_line_start"]:
            it["parse_uncertain"] = True
        if it["answer_line_start"] is not None and it["answer_line_end"] is not None:
            if it["answer_line_end"] < it["answer_line_start"]:
                it["parse_uncertain"] = True
        cleaned.append(it)
    return cleaned


def pick_auditor_system(tag: str) -> str:
    if tag == "math":
        return MATH_AUDITOR_SYSTEM
    if tag == "science":
        return SCIENCE_AUDITOR_SYSTEM
    if tag == "language":
        return LANGUAGE_AUDITOR_SYSTEM
    return LANGUAGE_AUDITOR_SYSTEM


def auditor_verify(
    orch: OpenAIOrchestrator,
    auditor_system: str,
    question_text: str,
    provided_answer: str,
) -> Dict[str, Any]:
    user = (
        "Audit this Q/A pair.\n\n"
        "QUESTION (verbatim):\n"
        f"{question_text}\n\n"
        "PROVIDED ANSWER (verbatim):\n"
        f"{provided_answer}\n\n"
        "Return JSON only matching schema with fields verdict, correct_answer, reasoning, confidence, tags."
    )
    res = orch.call_json_agent(auditor_system, user, AUDITOR_OUTPUT_SCHEMA, temperature=0.15)
    return res.data


def crosscheck(
    orch: OpenAIOrchestrator,
    question_text: str,
    provided_answer: str,
    proposed: Dict[str, Any],
) -> Dict[str, Any]:
    user = (
        "Re-verify the item independently.\n\n"
        "QUESTION:\n"
        f"{question_text}\n\n"
        "PROVIDED ANSWER:\n"
        f"{provided_answer}\n\n"
        "PROPOSED AUDIT RESULT:\n"
        f"{json.dumps(proposed, ensure_ascii=False)}\n\n"
        "Return JSON only matching schema."
    )
    res = orch.call_json_agent(CROSSCHECK_SYSTEM, user, CROSSCHECK_OUTPUT_SCHEMA, temperature=0.1)
    return res.data


def quality_check(orch: OpenAIOrchestrator, corrected_text: str) -> Dict[str, Any]:
    user = (
        "Check corrected file formatting.\n"
        "Rules:\n"
        "- Questions must be unchanged.\n"
        "- Answers must be labeled only using one of:\n"
        "  Answer (Verified):\n"
        "  Answer (Corrected):\n"
        "  Answer (Unverified):\n"
        "  Answer (Needs External Source):\n"
        "- No extra commentary.\n\n"
        "CORRECTED FILE CONTENT:\n"
        f"{corrected_text}"
    )
    res = orch.call_json_agent(QUALITY_SYSTEM, user, QUALITY_OUTPUT_SCHEMA, temperature=0.0)
    return res.data


def build_answer_line(verdict: str, provided: str, correct: Optional[str]) -> str:
    if verdict == "verified":
        return f"Answer (Verified): {provided.strip()}"
    if verdict == "corrected":
        return f"Answer (Corrected): {(correct or '').strip()}"
    if verdict == "needs_external_source":
        suffix = provided.strip() if provided.strip() else ""
        return f"Answer (Needs External Source): {suffix}".rstrip()
    suffix = provided.strip() if provided.strip() else ""
    return f"Answer (Unverified): {suffix}".rstrip()


def apply_replacements(lines: List[str], replacements: List[Tuple[int, int, List[str]]]) -> List[str]:
    out = lines[:]
    for start, end, new_lines in sorted(replacements, key=lambda x: x[0], reverse=True):
        if start == end + 1:
            out[start:start] = new_lines
        else:
            out[start:end + 1] = new_lines
    return out


def audit_one_file(
    orch: OpenAIOrchestrator,
    input_path: Path,
    input_root: Path,
    audited_root: Path,
    reports_root: Path,
    crosscheck_conf_threshold: float = 0.78,
) -> Tuple[Path, Path, Dict[str, Any]]:
    text = read_text(input_path)
    lines = text.splitlines()

    items = parser_extract_items(orch, text)

    audit_items: List[Dict[str, Any]] = []
    replacements: List[Tuple[int, int, List[str]]] = []

    for idx, it in enumerate(items, start=1):
        qid = str(it["question_id"] or str(idx))
        question_text = it["question_text"]
        provided_answer = it["provided_answer_text"] or ""

        tag = detect_tag(question_text)
        auditor_system = pick_auditor_system(tag)

        if it["parse_uncertain"] and (it["answer_line_start"] is None or it["answer_line_end"] is None):
            verdict = "unverified"
            correct_answer = None
            reasoning = "Parsing uncertain; could not reliably locate provided answer span to verify."
            confidence = 0.35
            tags = [tag if tag in ["math", "science", "language"] else "other"]
        else:
            proposed = auditor_verify(orch, auditor_system, question_text, provided_answer)
            if proposed["verdict"] == "corrected" or float(proposed["confidence"]) < crosscheck_conf_threshold:
                final = crosscheck(orch, question_text, provided_answer, proposed)
            else:
                final = proposed

            verdict = final["verdict"]
            correct_answer = final["correct_answer"]
            reasoning = final["reasoning"]
            confidence = float(final["confidence"])
            tags = final["tags"]

        ans_line = build_answer_line(verdict, provided_answer, correct_answer)

        indent = ""
        astart = it["answer_line_start"]
        aend = it["answer_line_end"]
        qend = it["question_line_end"]

        if astart is not None and aend is not None:
            s0 = astart - 1
            e0 = aend - 1
            if 0 <= s0 < len(lines):
                m = re.match(r"^(\s*)", lines[s0])
                indent = m.group(1) if m else ""
            replacements.append((s0, e0, [indent + ans_line]))
        else:
            insert_at = min(max(qend, 1), len(lines))
            replacements.append((insert_at, insert_at - 1, [ans_line]))

        audit_items.append(
            {
                "question_id": qid,
                "question_excerpt": normalize_excerpt(question_text, 160),
                "provided_answer": provided_answer,
                "verdict": verdict,
                "correct_answer": correct_answer if verdict == "corrected" else None,
                "reasoning": reasoning,
                "confidence": confidence,
                "tags": tags,
            }
        )

    corrected_lines = apply_replacements(lines, replacements)
    corrected_text = "\n".join(corrected_lines).rstrip() + "\n"

    qc = quality_check(orch, corrected_text)
    qc_issues = qc.get("issues", [])

    rel = input_path.relative_to(input_root)

    # IMPORTANT: exact same filename, exact same relative path under audited_output/
    out_path = audited_root / rel

    report_path = reports_root / rel.parent / (rel.stem + "_audit.json")

    stats = {
        "total_items": len(audit_items),
        "verified": sum(1 for x in audit_items if x["verdict"] == "verified"),
        "corrected": sum(1 for x in audit_items if x["verdict"] == "corrected"),
        "unverified": sum(1 for x in audit_items if x["verdict"] == "unverified"),
        "needs_external_source": sum(1 for x in audit_items if x["verdict"] == "needs_external_source"),
    }

    report_obj: Dict[str, Any] = {
        "input_file": str(input_path.as_posix()),
        "output_file": str(out_path.as_posix()),
        "items": audit_items,
        "stats": stats,
    }

    jsonschema_validate(instance=report_obj, schema=PER_FILE_REPORT_SCHEMA)

    write_text(out_path, corrected_text)

    safe_mkdir(report_path.parent)
    report_path.write_text(json.dumps(report_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    if qc_issues:
        qc_path = report_path.with_name(rel.stem + "_audit_qc_issues.json")
        qc_payload = {
            "input_file": str(input_path.as_posix()),
            "qc_status": qc.get("status", "fail"),
            "issues": qc_issues,
        }
        qc_path.write_text(json.dumps(qc_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_path, report_path, report_obj


def build_master_summary(per_file_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_files = len(per_file_reports)
    total_questions = sum(r["stats"]["total_items"] for r in per_file_reports)

    counts = {
        "verified": sum(r["stats"]["verified"] for r in per_file_reports),
        "corrected": sum(r["stats"]["corrected"] for r in per_file_reports),
        "unverified": sum(r["stats"]["unverified"] for r in per_file_reports),
        "needs_external_source": sum(r["stats"]["needs_external_source"] for r in per_file_reports),
    }

    def pct(x: int) -> float:
        return (x / total_questions) if total_questions else 0.0

    cat_counts: Dict[str, int] = {}
    for r in per_file_reports:
        for it in r["items"]:
            v = it["verdict"]
            if v == "verified":
                continue
            cat = error_category_from_reasoning(it.get("reasoning", ""), v)
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    non_verified = max(1, total_questions - counts["verified"])
    common_error_categories = sorted(
        [
            {
                "category": k,
                "count": v,
                "percent_of_nonverified": (v / non_verified),
            }
            for k, v in cat_counts.items()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )

    file_rates = []
    for r in per_file_reports:
        t = r["stats"]["total_items"]
        c = r["stats"]["corrected"]
        file_rates.append(
            {
                "input_file": r["input_file"],
                "corrected": c,
                "total": t,
                "correction_rate": (c / t) if t else 0.0,
            }
        )
    file_rates_sorted = sorted(file_rates, key=lambda x: x["correction_rate"], reverse=True)[:20]

    return {
        "total_files_processed": total_files,
        "total_questions_audited": total_questions,
        "counts": counts,
        "percentages": {k: pct(v) for k, v in counts.items()},
        "most_common_error_categories": common_error_categories[:15],
        "files_with_highest_correction_rate": file_rates_sorted,
    }


def enumerate_txt_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.txt") if p.is_file()])


def main() -> int:
    ap = argparse.ArgumentParser(description="auditor: Audit and correct Q/A .txt files under output/")
    ap.add_argument("--input-root", default="output", help="Root directory containing .txt files (default: output)")
    ap.add_argument("--audited-root", default="audited_output", help="Root directory for corrected files")
    ap.add_argument("--reports-root", default="audit_reports", help="Root directory for audit reports")
    ap.add_argument("--limit-files", type=int, default=0, help="Limit number of files processed (0 = no limit)")
    args = ap.parse_args()

    input_root = Path(args.input_root).resolve()
    audited_root = Path(args.audited_root).resolve()
    reports_root = Path(args.reports_root).resolve()

    if not input_root.exists() or not input_root.is_dir():
        print(f"ERROR: input root not found or not a directory: {input_root}")
        return 2

    safe_mkdir(audited_root)
    safe_mkdir(reports_root)

    files = enumerate_txt_files(input_root)
    if args.limit_files and args.limit_files > 0:
        files = files[: args.limit_files]

    if not files:
        print(f"No .txt files found under {input_root}")
        return 0

    orch = OpenAIOrchestrator()

    per_file_reports: List[Dict[str, Any]] = []
    failures: List[str] = []

    for i, fpath in enumerate(files, start=1):
        rel = fpath.relative_to(input_root)
        print(f"[{i}/{len(files)}] auditor auditing: {rel.as_posix()}", flush=True)
        try:
            out_path, rep_path, rep_obj = audit_one_file(
                orch=orch,
                input_path=fpath,
                input_root=input_root,
                audited_root=audited_root,
                reports_root=reports_root,
            )
            per_file_reports.append(rep_obj)
            print(f"  -> corrected: {out_path.relative_to(audited_root).as_posix()}", flush=True)
            print(f"  -> report:    {rep_path.relative_to(reports_root).as_posix()}", flush=True)
        except Exception as e:
            msg = f"{rel.as_posix()}: {e}"
            failures.append(msg)
            print(f"  !! FAILED: {msg}", flush=True)

    summary = build_master_summary(per_file_reports)
    summary["failures"] = failures

    master_path = reports_root / "MASTER_SUMMARY.json"
    write_text(master_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print("\nauditor complete.")
    print(f"Corrected files root: {audited_root}")
    print(f"Audit reports root:   {reports_root}")
    print(f"Master summary:       {master_path}")
    if failures:
        print(f"Failures: {len(failures)} (see MASTER_SUMMARY.json)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
