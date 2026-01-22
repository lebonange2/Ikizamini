#!/usr/bin/env python3
"""
ikizamini_local.py

IKIZAMINI Local Web UI (Flask) + Ollama with long-running stability + persistence.

Key properties:
- Runs generation in background thread (UI doesn't timeout).
- Persists job state and objective outputs to SQLite + disk as it runs.
- Partial recovery: even if process crashes, already-written outputs remain downloadable.
- Avoids BrokenPipe crashes by logging to files and catching stdout errors.

Run:
  python3 ikizamini_local.py
Open:
  http://127.0.0.1:8000  (or Runpod public URL)
"""

from __future__ import annotations

import io
import os
import re
import json
import time
import uuid
import hashlib
import random
import zipfile
import sys
import unicodedata
import threading
import sqlite3
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, request, redirect, url_for, send_file, render_template_string, abort, jsonify
from jsonschema import validate as jsonschema_validate


# =============================================================================
# Config
# =============================================================================

DATA_DIR = os.environ.get("IKIZAMINI_DATA_DIR", "./ikizamini_data")
DB_PATH = os.path.join(DATA_DIR, "ikizamini.db")
JOBS_DIR = os.path.join(DATA_DIR, "jobs")

DEFAULT_TIMEOUT_S = int(os.environ.get("IKIZAMINI_OLLAMA_TIMEOUT", "600"))  # long default
DEFAULT_MAX_RETRIES = int(os.environ.get("IKIZAMINI_OLLAMA_RETRIES", "4"))
HEARTBEAT_EVERY_SEC = 10

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)

app = Flask(__name__)

# Background thread coordination
RUNNER_LOCK = threading.Lock()  # prevents accidentally starting multiple runners for same job concurrently


# =============================================================================
# DB Helpers
# =============================================================================

def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def db_init() -> None:
    conn = db_connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
      job_id TEXT PRIMARY KEY,
      status TEXT NOT NULL,
      created_at_unix INTEGER NOT NULL,
      started_at_unix INTEGER,
      finished_at_unix INTEGER,
      heartbeat_unix INTEGER,
      input_filename TEXT,
      input_path TEXT,
      config_json TEXT NOT NULL,
      requested INTEGER DEFAULT 0,
      completed INTEGER DEFAULT 0,
      failures_json TEXT DEFAULT '[]',
      master_path TEXT,
      log_path TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS objectives (
      job_id TEXT NOT NULL,
      objective_id TEXT NOT NULL,
      objective_text TEXT NOT NULL,
      path_json TEXT NOT NULL,
      status TEXT NOT NULL,
      started_at_unix INTEGER,
      finished_at_unix INTEGER,
      error_text TEXT,
      output_txt_path TEXT,
      output_json_path TEXT,
      PRIMARY KEY(job_id, objective_id),
      FOREIGN KEY(job_id) REFERENCES jobs(job_id)
    );
    """)

    conn.commit()
    conn.close()


def db_mark_stale_jobs_as_interrupted(stale_after_sec: int = 3600) -> None:
    """If server crashed, jobs may be stuck in running/queued. Mark as interrupted if heartbeat is old."""
    now = int(time.time())
    conn = db_connect()
    cur = conn.cursor()

    cur.execute("""
      SELECT job_id, status, heartbeat_unix
      FROM jobs
      WHERE status IN ('queued','running')
    """)
    rows = cur.fetchall()

    for r in rows:
        hb = r["heartbeat_unix"] or 0
        if hb and (now - hb) > stale_after_sec:
            cur.execute("UPDATE jobs SET status=? WHERE job_id=?", ("interrupted", r["job_id"]))
        elif not hb and (now - (r["created_at_unix"] or now)) > stale_after_sec:
            cur.execute("UPDATE jobs SET status=? WHERE job_id=?", ("interrupted", r["job_id"]))

    conn.commit()
    conn.close()


# =============================================================================
# Logging (file-based + safe stdout)
# =============================================================================

def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_print(line: str) -> None:
    """Print without crashing if stdout is closed (BrokenPipe)."""
    try:
        print(line, flush=True)
    except BrokenPipeError:
        # stdout closed (SSH disconnect); ignore
        pass
    except Exception:
        pass


def append_job_log(job_id: str, message: str) -> None:
    """Append line to job log file and try to print to stdout safely."""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT log_path FROM jobs WHERE job_id=?", (job_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return

    log_path = row["log_path"]
    if not log_path:
        return

    line = f"[{_now_ts()}] {message}"
    # Write to file (durable)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # If file writing fails, we still try stdout
        pass

    safe_print(line)


def read_log_tail(log_path: str, max_lines: int = 120) -> List[str]:
    """Read last N lines from a potentially large file efficiently."""
    if not log_path or not os.path.exists(log_path):
        return []
    try:
        # Simple approach: read from end in chunks
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= max_lines:
                step = min(block, size)
                f.seek(size - step)
                data = f.read(step) + data
                size -= step
            lines = data.splitlines()[-max_lines:]
            return [ln.decode("utf-8", errors="replace") for ln in lines]
    except Exception:
        return []


# =============================================================================
# Outline parsing
# =============================================================================

@dataclass
class ObjNode:
    obj_id: str
    title: str
    indent: int
    children: List["ObjNode"] = field(default_factory=list)


OBJ_LINE_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.*\S)\s*$")

UNICODE_SPACES = [
    "\u00A0", "\u2007", "\u202F", "\u205F", "\u3000",
    "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005",
    "\u2006", "\u2008", "\u2009", "\u200A",
]


def normalize_outline_text(text: str) -> str:
    for sp in UNICODE_SPACES:
        text = text.replace(sp, " ")
    text = text.replace("\t", "    ")
    text = "\n".join([ln.rstrip() for ln in text.splitlines()])
    return text


def parse_objectives(text: str) -> List[ObjNode]:
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
        indent = len(raw) - len(raw.lstrip(" "))

        node = ObjNode(obj_id=obj_id, title=title, indent=indent)

        while stack and indent <= stack[-1].indent:
            stack.pop()

        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)

        stack.append(node)

    return roots


def id_depth(obj_id: str) -> int:
    return len([p for p in obj_id.split(".") if p.strip()])


def is_learning_objective_id(obj_id: str) -> bool:
    # Only IDs like 1.1.1.1+ are learning objectives
    return id_depth(obj_id) >= 4


def collect_learning_objectives_with_paths(roots: List[ObjNode]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def dfs(n: ObjNode, path: List[Tuple[str, str]]):
        new_path = path + [(n.obj_id, n.title)]
        if is_learning_objective_id(n.obj_id):
            out.append({"objective_id": n.obj_id, "objective_text": n.title, "path": new_path})
        else:
            for c in n.children:
                dfs(c, new_path)

    for r in roots:
        dfs(r, [])
    return out


# =============================================================================
# JSON extraction + normalization helpers
# =============================================================================

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def extract_json_value(text: str) -> Any:
    m = JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            val, _end = decoder.raw_decode(text[i:])
            return val
        except json.JSONDecodeError:
            continue

    raise ValueError("No JSON value found in model output.")


def _normalize_choice_letter(x: Any) -> str:
    s = str(x or "").strip().upper()
    if s in {"A", "B", "C", "D"}:
        return s
    m = re.search(r"\b([ABCD])\b", s)
    return (m.group(1) if m else "A")


_CONFUSABLE_TRANSLATION = str.maketrans({
    "\u0430": "a", "\u0435": "e", "\u043E": "o", "\u0440": "p",
    "\u0441": "c", "\u0445": "x", "\u0456": "i", "\u0458": "j",
})


def _canonical_key(key: Any) -> str:
    s = str(key or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_CONFUSABLE_TRANSLATION)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _map_question_keys(q: Dict[str, Any]) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {}
    for k, v in q.items():
        ck = _canonical_key(k)
        if ck in {"id", "qid", "questionid"}:
            mapped["id"] = v
        elif ck in {"stem", "question", "prompt"}:
            mapped["stem"] = v
        elif ck in {"choices", "answerchoices", "answerchoice", "options", "answers"}:
            mapped["choices"] = v
        elif ck in {"correctchoice", "correctanswer", "correct"}:
            mapped["correct_choice"] = v
        elif ck in {"solution", "workedsolution", "worked"}:
            mapped["solution"] = v
        elif ck in {"explanation", "rationale"}:
            mapped["explanation"] = v
        elif ck in {"commonmistakenotes", "commonmistake", "commonmistakenote"}:
            mapped["common_mistake_notes"] = v
    return mapped


def sanitize_question_list(questions_in: Any) -> List[Dict[str, Any]]:
    if not isinstance(questions_in, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, q in enumerate(questions_in[:7], start=1):
        if not isinstance(q, dict):
            continue
        qn = _map_question_keys(q)

        ch_raw = qn.get("choices", {})
        choices: Dict[str, str] = {"A": "", "B": "", "C": "", "D": ""}
        if isinstance(ch_raw, dict):
            for kk in ["A", "B", "C", "D"]:
                choices[kk] = str(ch_raw.get(kk, "")).strip()
        elif isinstance(ch_raw, list) and len(ch_raw) >= 4:
            choices["A"] = str(ch_raw[0]).strip()
            choices["B"] = str(ch_raw[1]).strip()
            choices["C"] = str(ch_raw[2]).strip()
            choices["D"] = str(ch_raw[3]).strip()

        out.append({
            "id": str(qn.get("id") or f"Q{idx}").strip(),
            "stem": str(qn.get("stem") or "").strip(),
            "choices": choices,
            "correct_choice": _normalize_choice_letter(qn.get("correct_choice")),
            "solution": str(qn.get("solution") or "").strip(),
            "explanation": str(qn.get("explanation") or "").strip(),
            "common_mistake_notes": str(qn.get("common_mistake_notes") or "").strip(),
        })
    return out


def sanitize_ikizamini(candidate: Dict[str, Any], objective_id: str, objective_text: str) -> Dict[str, Any]:
    difficulty_profile = candidate.get("difficulty_profile", "mixed") if isinstance(candidate, dict) else "mixed"
    questions_out = sanitize_question_list(candidate.get("questions", []) if isinstance(candidate, dict) else [])
    return {
        "objective_id": objective_id,
        "objective_text": objective_text,
        "difficulty_profile": str(difficulty_profile or "mixed"),
        "questions": questions_out,
    }


def coerce_ikizamini(value: Any, objective_id: str, objective_text: str) -> Dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get("questions"), list):
        return sanitize_ikizamini(value, objective_id, objective_text)

    if isinstance(value, dict):
        for k in ("questions", "items", "data", "revised_questions"):
            if isinstance(value.get(k), list):
                return sanitize_ikizamini({"questions": value[k]}, objective_id, objective_text)

    if isinstance(value, list):
        if len(value) < 7:
            raise ValueError(f"Expected at least 7 questions; got {len(value)}.")
        return sanitize_ikizamini({"questions": value}, objective_id, objective_text)

    raise ValueError("Model output was not an object with questions nor a list of questions.")


# =============================================================================
# Schemas
# =============================================================================

IKIZAMINI_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "objective_id": {"type": "string"},
        "objective_text": {"type": "string"},
        "difficulty_profile": {"type": "string"},
        "questions": {
            "type": "array", "minItems": 7, "maxItems": 7,
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
                "required": ["id", "stem", "choices", "correct_choice", "solution", "explanation", "common_mistake_notes"],
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
            "type": "array", "minItems": 7, "maxItems": 7,
            "items": IKIZAMINI_SCHEMA["properties"]["questions"]["items"],
        },
        "score": {
            "type": "object", "additionalProperties": False,
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


def validate_ikizamini(obj: Dict[str, Any]) -> None:
    jsonschema_validate(instance=obj, schema=IKIZAMINI_SCHEMA)


def validate_manager(obj: Dict[str, Any]) -> None:
    jsonschema_validate(instance=obj, schema=MANAGER_SCHEMA)


# =============================================================================
# Prompts
# =============================================================================

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

Return ONLY JSON that matches the schema. No extra keys. No commentary.
"""

WORKER_REPAIR_SYSTEM = """You are workerAgent in REPAIR mode.
Fix the provided set so it:
- matches the JSON schema exactly (no extra keys, no missing keys),
- has correct math and solutions,
- has exactly ONE correct choice per question,
- has 7 distinct questions aligned to the objective,
- includes solution + explanation + common mistake notes for each question.

Return ONLY JSON that matches the schema. No extra keys. No commentary.
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

Return ONLY JSON with keys: status, issues, revised_questions, score.
No objective_id/objective_text/questions/difficulty_profile keys.
"""

MANAGER_FIX_SYSTEM = """You MUST output valid JSON matching the manager schema EXACTLY.

Do not include keys like objective_id/objective_text/questions/difficulty_profile.
Only include: status, issues, revised_questions, score.
Return ONLY JSON (no markdown, no extra text).
"""


# =============================================================================
# Ollama calls (robust)
# =============================================================================

def backoff_sleep(attempt: int) -> None:
    base = min(2 ** attempt, 30)
    jitter = random.uniform(0.0, 0.8)
    time.sleep(base + jitter)


def ollama_chat(
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    num_ctx: int,
    timeout_s: int,
    format_spec: Any = "json",
) -> str:
    url = base_url.rstrip("/") + "/api/chat"

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "format": format_spec,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }

    # Important: use a tuple timeout to avoid stalls (connect, read)
    # connect timeout shorter, read timeout long
    timeout = (10, timeout_s)

    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code >= 400 and isinstance(format_spec, dict):
        # Some setups reject schema dict; fallback to "json"
        payload["format"] = "json"
        r = requests.post(url, json=payload, timeout=timeout)

    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}) or {}).get("content", "").strip()


def call_ollama_json(
    base_url: str,
    model: str,
    system: str,
    user: str,
    validate_fn,
    temperature: float,
    num_ctx: int,
    timeout_s: int,
    max_retries: int,
    coerce_fn=None,
    format_spec: Any = "json",
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    last_raw: Optional[str] = None

    for attempt in range(max_retries):
        try:
            raw = ollama_chat(
                base_url=base_url,
                model=model,
                system=system,
                user=user,
                temperature=temperature,
                num_ctx=num_ctx,
                timeout_s=timeout_s,
                format_spec=format_spec,
            )
            last_raw = raw
            val = extract_json_value(raw)
            if coerce_fn is not None:
                val = coerce_fn(val)
            validate_fn(val)
            return val
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as e:
            last_err = e
            backoff_sleep(attempt)
        except requests.exceptions.ConnectionError as e:
            last_err = e
            backoff_sleep(attempt)
        except Exception as e:
            last_err = e
            backoff_sleep(attempt)

    preview = ""
    if last_raw:
        preview = last_raw[:800] + ("..." if len(last_raw) > 800 else "")
    raise RuntimeError(f"Ollama call failed after retries. Last error: {last_err}. Raw preview: {preview}")


# =============================================================================
# Quality checks
# =============================================================================

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
            issues.append(f"Q{i}: not an object.")
            continue
        ch = q.get("choices", {})
        if not isinstance(ch, dict):
            issues.append(f"Q{i}: choices is not an object.")
            continue
        vals = [str(ch.get(k, "")).strip() for k in ["A", "B", "C", "D"]]
        if len(set(vals)) != 4:
            issues.append(f"Q{i}: duplicate choice text detected (A/B/C/D must be distinct).")
        cc = q.get("correct_choice")
        if cc not in {"A", "B", "C", "D"}:
            issues.append(f"Q{i}: correct_choice is invalid: {cc}")
        for k in ["solution", "explanation", "common_mistake_notes"]:
            if not str(q.get(k, "")).strip():
                issues.append(f"Q{i}: missing/empty field: {k}")
    return issues


# =============================================================================
# Worker/Manager orchestration
# =============================================================================

def worker_generate(base_url: str, model: str, objective_id: str, objective_text: str, num_ctx: int,
                    timeout_s: int, max_retries: int) -> Dict[str, Any]:
    user = f"""Learning objective:
- id: {objective_id}
- text: {objective_text}

Generate EXACTLY 7 distinct multiple-choice questions.

Return ONLY a JSON object with keys:
objective_id, objective_text, difficulty_profile, questions

No extra keys.
"""
    return call_ollama_json(
        base_url=base_url,
        model=model,
        system=WORKER_SYSTEM,
        user=user,
        validate_fn=validate_ikizamini,
        temperature=0.25,
        num_ctx=num_ctx,
        timeout_s=timeout_s,
        max_retries=max_retries,
        coerce_fn=lambda v: coerce_ikizamini(v, objective_id, objective_text),
        format_spec=IKIZAMINI_SCHEMA,
    )


def worker_repair(base_url: str, model: str, objective_id: str, objective_text: str,
                  candidate: Dict[str, Any], issues: List[str], num_ctx: int,
                  timeout_s: int, max_retries: int) -> Dict[str, Any]:
    user = f"""Learning objective:
- id: {objective_id}
- text: {objective_text}

Issues to fix:
{json.dumps(issues, ensure_ascii=False, indent=2)}

Current JSON:
{json.dumps(candidate, ensure_ascii=False)}

Return ONLY corrected JSON matching the schema, with exactly 7 questions.
No extra keys.
"""
    return call_ollama_json(
        base_url=base_url,
        model=model,
        system=WORKER_REPAIR_SYSTEM,
        user=user,
        validate_fn=validate_ikizamini,
        temperature=0.2,
        num_ctx=num_ctx,
        timeout_s=timeout_s,
        max_retries=max_retries,
        coerce_fn=lambda v: coerce_ikizamini(v, objective_id, objective_text),
        format_spec=IKIZAMINI_SCHEMA,
    )


def sanitize_manager_output(value: Any, fallback_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get("questions"), list):
        rq = sanitize_question_list(value.get("questions"))
        return {
            "status": "pass",
            "issues": ["Manager returned IKIZAMINI-shaped output; treated as pass."],
            "revised_questions": rq if len(rq) == 7 else fallback_questions,
            "score": {"relevance": 4, "distinctness": 4, "unique_solution": 4, "explanations": 4},
        }

    if isinstance(value, list):
        rq = sanitize_question_list(value)
        return {
            "status": "pass",
            "issues": ["Manager returned bare list; treated as revised_questions."],
            "revised_questions": rq if len(rq) == 7 else fallback_questions,
            "score": {"relevance": 4, "distinctness": 4, "unique_solution": 4, "explanations": 4},
        }

    if not isinstance(value, dict):
        return {
            "status": "fail",
            "issues": ["Manager returned non-object output; using fallback questions."],
            "revised_questions": fallback_questions,
            "score": {"relevance": 2, "distinctness": 2, "unique_solution": 2, "explanations": 2},
        }

    status = value.get("status", "fail")
    if status not in {"pass", "fail"}:
        status = "fail"

    issues = value.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]

    rq_raw = value.get("revised_questions", value.get("questions", []))
    rq = sanitize_question_list(rq_raw)

    score = value.get("score", {})
    if not isinstance(score, dict):
        score = {}

    def _clamp_int(x, default):
        try:
            v = int(x)
        except Exception:
            v = default
        return max(0, min(5, v))

    score_out = {
        "relevance": _clamp_int(score.get("relevance", 3), 3),
        "distinctness": _clamp_int(score.get("distinctness", 3), 3),
        "unique_solution": _clamp_int(score.get("unique_solution", 3), 3),
        "explanations": _clamp_int(score.get("explanations", 3), 3),
    }

    return {
        "status": status,
        "issues": [str(x) for x in issues],
        "revised_questions": rq if len(rq) == 7 else fallback_questions,
        "score": score_out,
    }


def manager_review(base_url: str, model: str, objective_id: str, objective_text: str,
                   candidate: Dict[str, Any], num_ctx: int, timeout_s: int, max_retries: int) -> Dict[str, Any]:
    fallback_questions = sanitize_question_list(candidate.get("questions", []))

    base_user = f"""Learning objective:
- id: {objective_id}
- text: {objective_text}

Candidate JSON:
{json.dumps(candidate, ensure_ascii=False)}

Return ONLY JSON with keys: status, issues, revised_questions, score.
"""

    last_err: Optional[str] = None
    last_raw: Optional[str] = None

    for attempt in range(3):
        user = base_user if attempt == 0 else f"""Your previous output did not match the manager schema.

Validation error:
{last_err}

Your previous output:
{last_raw}

Now return ONLY valid JSON matching exactly:
{json.dumps(MANAGER_SCHEMA, ensure_ascii=False)}

Remember: keys must be ONLY status, issues, revised_questions, score.
No objective_id/objective_text/questions/difficulty_profile keys.
"""
        system = MANAGER_SYSTEM if attempt == 0 else MANAGER_FIX_SYSTEM

        try:
            raw = ollama_chat(
                base_url=base_url,
                model=model,
                system=system,
                user=user,
                temperature=0.1,
                num_ctx=num_ctx,
                timeout_s=timeout_s,
                format_spec=MANAGER_SCHEMA,
            )
            last_raw = raw
            val = extract_json_value(raw)
            val2 = sanitize_manager_output(val, fallback_questions)
            validate_manager(val2)
            return val2
        except Exception as e:
            last_err = str(e)[:1200]
            backoff_sleep(attempt)

    # Safe fallback
    val_fallback = {
        "status": "pass",
        "issues": ["Manager failed schema compliance after retries; used candidate questions unchanged."],
        "revised_questions": fallback_questions,
        "score": {"relevance": 3, "distinctness": 3, "unique_solution": 3, "explanations": 3},
    }
    validate_manager(val_fallback)
    return val_fallback


def generate_for_objective_strict(
    base_url: str,
    worker_model: str,
    manager_model: str,
    objective_id: str,
    objective_text: str,
    max_rounds: int,
    num_ctx: int,
    timeout_s: int,
    max_retries: int,
) -> Dict[str, Any]:
    candidate = worker_generate(
        base_url, worker_model, objective_id, objective_text,
        num_ctx=num_ctx, timeout_s=timeout_s, max_retries=max_retries
    )

    for _ in range(max_rounds):
        issues = local_issues(candidate)
        if issues:
            candidate = worker_repair(
                base_url, worker_model, objective_id, objective_text,
                candidate, issues, num_ctx=num_ctx, timeout_s=timeout_s, max_retries=max_retries
            )
            continue

        review = manager_review(
            base_url, manager_model, objective_id, objective_text,
            candidate, num_ctx=num_ctx, timeout_s=timeout_s, max_retries=max_retries
        )

        candidate = {
            "objective_id": objective_id,
            "objective_text": objective_text,
            "difficulty_profile": candidate.get("difficulty_profile", "mixed") or "mixed",
            "questions": review["revised_questions"],
        }
        validate_ikizamini(candidate)

        if review["status"] == "pass":
            return candidate

        mgr_issues = review.get("issues", []) or ["Manager marked fail without detailed issues."]
        candidate = worker_repair(
            base_url, worker_model, objective_id, objective_text,
            candidate, mgr_issues, num_ctx=num_ctx, timeout_s=timeout_s, max_retries=max_retries
        )

    raise RuntimeError(f"Could not reach manager approval after {max_rounds} rounds for objective {objective_id}.")


# =============================================================================
# Output formatting
# =============================================================================

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
    return f"{objective_id}_{safe_filename(objective_text)}_{suffix}"


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


def objective_file_text(path: List[Tuple[str, str]], objective_id: str, objective_text: str,
                        ikizamini_set: Optional[Dict[str, Any]], error: Optional[str]) -> str:
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(f"LEARNING OBJECTIVE: {objective_id} {objective_text}")
    lines.append("=" * 78)

    if len(path) > 1:
        breadcrumb = " > ".join([f"{pid} {ptitle}" for pid, ptitle in path[:-1]])
        lines.append(f"Context: {breadcrumb}")
        lines.append("-" * 78)

    if ikizamini_set is not None:
        lines.append(f"Difficulty profile: {ikizamini_set.get('difficulty_profile','mixed')}")
        lines.append("-" * 78)
        lines.append(format_questions_block(ikizamini_set))
    else:
        lines.append("STATUS: FAILED TO GENERATE QUESTIONS FOR THIS OBJECTIVE")
        lines.append("-" * 78)
        lines.append((error or "Unknown error.").strip())
        lines.append("\nSuggested action: increase max rounds / timeout or switch models.\n")

    return "\n".join(lines).rstrip() + "\n"


def write_master_incremental(input_text: str, roots: List[ObjNode], outputs_by_id: Dict[str, str], meta: Dict[str, Any]) -> str:
    """Build a master file from the outline tree and any available per-objective .txt blocks."""
    def heading_line(node: ObjNode) -> str:
        return f"{node.obj_id} {node.title}"

    def render(node: ObjNode, level: int) -> List[str]:
        indent = "   " * max(level - 1, 0)
        out: List[str] = [f"{indent}{heading_line(node)}"]

        if is_learning_objective_id(node.obj_id):
            block = outputs_by_id.get(node.obj_id)
            if block:
                # indent the block
                out.append(f"{indent}{'-' * 78}")
                out.append(block.replace("\n", "\n" + indent).rstrip())
            else:
                out.append(f"{indent}(No generated output for this objective yet.)")
        else:
            for c in node.children:
                out.extend(render(c, level + 1))
        return out

    lines: List[str] = []
    lines.append("IKIZAMINI MASTER OUTPUT (Durable)")
    lines.append(f"Worker model: {meta.get('worker_model','')}")
    lines.append(f"Manager model: {meta.get('manager_model','')}")
    lines.append(f"Ollama URL: {meta.get('ollama_url','')}")
    lines.append(f"Generated at (unix): {meta.get('generated_at_unix','')}")
    lines.append(f"Objectives completed: {meta.get('completed',0)} / {meta.get('requested',0)}")
    lines.append("\n" + "#" * 78 + "\n")

    for r in roots:
        lines.extend(render(r, 1))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# =============================================================================
# Persistence: job dir layout
# =============================================================================

def job_dir(job_id: str) -> str:
    p = os.path.join(JOBS_DIR, job_id)
    os.makedirs(p, exist_ok=True)
    return p


def outputs_dir(job_id: str) -> str:
    p = os.path.join(job_dir(job_id), "outputs")
    os.makedirs(p, exist_ok=True)
    return p


def ensure_job_files(job_id: str) -> Tuple[str, str]:
    """Return (input_path, log_path). Create log file if missing."""
    jd = job_dir(job_id)
    inp = os.path.join(jd, "input.txt")
    logp = os.path.join(jd, "job.log")
    if not os.path.exists(logp):
        with open(logp, "a", encoding="utf-8") as f:
            f.write(f"[{_now_ts()}] Log created for job {job_id}\n")
    return inp, logp


def write_output_files(job_id: str, obj_id: str, obj_text: str, path: List[Tuple[str, str]],
                       ik_set: Optional[Dict[str, Any]], error: Optional[str]) -> Tuple[str, str]:
    """Write per-objective .txt and .json (if present). Return (txt_path, json_path_or_empty)."""
    base = unique_objective_filename(obj_id, obj_text)
    od = outputs_dir(job_id)
    txt_path = os.path.join(od, base + ".txt")
    json_path = os.path.join(od, base + ".json")

    txt = objective_file_text(path, obj_id, obj_text, ik_set, error)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt)

    if ik_set is not None:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ik_set, f, ensure_ascii=False, indent=2)
        return txt_path, json_path

    # If failed, still write a json file? Keep empty to indicate none.
    return txt_path, ""


def build_zip_from_outputs(job_id: str) -> bytes:
    """Zip whatever outputs exist right now (partial allowed)."""
    od = outputs_dir(job_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # include outputs directory contents
        for root, _dirs, files in os.walk(od):
            for fn in files:
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, job_dir(job_id))
                zf.write(full, arcname=arc)

        # include index.txt built from DB
        index = build_index_text(job_id)
        zf.writestr("outputs/index.txt", index)

        # include master if exists or build partial master
        master = build_master_text(job_id)
        zf.writestr("master_partial.txt", master)

    return buf.getvalue()


def build_index_text(job_id: str) -> str:
    conn = db_connect()
    cur = conn.cursor()

    cur.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
    job = cur.fetchone()

    cur.execute("""
      SELECT objective_id, objective_text, status, output_txt_path
      FROM objectives
      WHERE job_id=?
      ORDER BY objective_id
    """, (job_id,))
    objs = cur.fetchall()
    conn.close()

    if not job:
        return "Job not found.\n"

    cfg = json.loads(job["config_json"])
    lines: List[str] = []
    lines.append("IKIZAMINI PER-OBJECTIVE INDEX (Durable)")
    lines.append(f"Job: {job_id}")
    lines.append(f"Status: {job['status']}")
    lines.append(f"Ollama URL: {cfg.get('ollama_url','')}")
    lines.append(f"Worker model: {cfg.get('worker_model','')}")
    lines.append(f"Manager model: {cfg.get('manager_model','')}")
    lines.append(f"Created at (unix): {job['created_at_unix']}")
    lines.append(f"Requested: {job['requested']}  Completed: {job['completed']}")
    lines.append("")
    for r in objs:
        st = r["status"]
        p = r["output_txt_path"] or ""
        lines.append(f"- {r['objective_id']} {r['objective_text']} [{st}] {p}")
    return "\n".join(lines).rstrip() + "\n"


def build_master_text(job_id: str) -> str:
    """Build partial or full master from input outline + per-objective txt content from disk."""
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
    job = cur.fetchone()
    conn.close()
    if not job:
        return "Job not found.\n"

    input_path = job["input_path"]
    if not input_path or not os.path.exists(input_path):
        return "Input file missing; cannot build master.\n"

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        input_text = f.read()

    roots = parse_objectives(input_text)

    # Load any available objective outputs: map objective_id -> txt content
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("""
      SELECT objective_id, output_txt_path
      FROM objectives
      WHERE job_id=? AND output_txt_path IS NOT NULL
    """, (job_id,))
    rows = cur.fetchall()
    conn.close()

    outputs_by_id: Dict[str, str] = {}
    for r in rows:
        p = r["output_txt_path"]
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    outputs_by_id[r["objective_id"]] = f.read()
            except Exception:
                pass

    cfg = json.loads(job["config_json"])
    meta = {
        "ollama_url": cfg.get("ollama_url", ""),
        "worker_model": cfg.get("worker_model", ""),
        "manager_model": cfg.get("manager_model", ""),
        "generated_at_unix": int(time.time()),
        "requested": job["requested"],
        "completed": job["completed"],
    }
    return write_master_incremental(input_text, roots, outputs_by_id, meta)


# =============================================================================
# Job runner (background)
# =============================================================================

def update_job_heartbeat(job_id: str) -> None:
    conn = db_connect()
    conn.execute("UPDATE jobs SET heartbeat_unix=? WHERE job_id=?", (int(time.time()), job_id))
    conn.commit()
    conn.close()


def run_job(job_id: str, resume: bool = False) -> None:
    """Background worker: generate objectives, persist after each objective."""
    with RUNNER_LOCK:
        conn = db_connect()
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
        timeout_s = int(cfg.get("timeout_s", DEFAULT_TIMEOUT_S))
        max_retries = int(cfg.get("max_retries", DEFAULT_MAX_RETRIES))

        input_path = job["input_path"]
        if not input_path or not os.path.exists(input_path):
            append_job_log(job_id, "ERROR: input file missing; cannot start.")
            conn = db_connect()
            conn.execute("UPDATE jobs SET status=?, finished_at_unix=? WHERE job_id=?",
                         ("error", int(time.time()), job_id))
            conn.commit()
            conn.close()
            return

        # Mark running
        conn = db_connect()
        conn.execute("UPDATE jobs SET status=?, started_at_unix=?, heartbeat_unix=? WHERE job_id=?",
                     ("running", job["started_at_unix"] or int(time.time()), int(time.time()), job_id))
        conn.commit()
        conn.close()

        append_job_log(job_id, "============================================================")
        append_job_log(job_id, "BACKGROUND GENERATION STARTED")
        append_job_log(job_id, f"Ollama URL: {ollama_url}")
        append_job_log(job_id, f"Worker model: {worker_model}")
        append_job_log(job_id, f"Manager model: {manager_model}")
        append_job_log(job_id, f"Max rounds: {max_rounds}  num_ctx: {num_ctx}")
        append_job_log(job_id, f"Timeout: {timeout_s}s  Retries: {max_retries}")
        append_job_log(job_id, "============================================================")

        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        roots = parse_objectives(content)
        objectives = collect_learning_objectives_with_paths(roots)
        if limit and limit > 0:
            objectives = objectives[:limit]

        # Ensure objectives exist in DB (idempotent)
        conn = db_connect()
        cur = conn.cursor()
        for obj in objectives:
            cur.execute("""
              INSERT OR IGNORE INTO objectives
              (job_id, objective_id, objective_text, path_json, status)
              VALUES (?, ?, ?, ?, ?)
            """, (job_id, obj["objective_id"], obj["objective_text"],
                  json.dumps(obj["path"], ensure_ascii=False), "pending"))
        conn.commit()
        conn.close()

        # Set requested if needed
        conn = db_connect()
        conn.execute("UPDATE jobs SET requested=? WHERE job_id=?", (len(objectives), job_id))
        conn.commit()
        conn.close()

        failures: List[str] = json.loads(job["failures_json"] or "[]")

        last_hb = time.time()

        for idx, obj in enumerate(objectives, start=1):
            # heartbeat
            if time.time() - last_hb >= HEARTBEAT_EVERY_SEC:
                update_job_heartbeat(job_id)
                last_hb = time.time()

            obj_id = obj["objective_id"]
            obj_text = obj["objective_text"]
            path = obj["path"]

            # If resuming, skip completed objectives
            if resume:
                conn = db_connect()
                cur = conn.cursor()
                cur.execute("""
                  SELECT status FROM objectives WHERE job_id=? AND objective_id=?
                """, (job_id, obj_id))
                row = cur.fetchone()
                conn.close()
                if row and row["status"] in ("done", "failed"):
                    # still count completion in job.completed later
                    continue

            append_job_log(job_id, f"[{idx}/{len(objectives)}] Processing: {obj_id} - {obj_text[:70]}...")

            conn = db_connect()
            conn.execute("""
              UPDATE objectives
              SET status=?, started_at_unix=?, error_text=NULL
              WHERE job_id=? AND objective_id=?
            """, ("running", int(time.time()), job_id, obj_id))
            conn.commit()
            conn.close()

            start = time.time()
            ik: Optional[Dict[str, Any]] = None
            err: Optional[str] = None

            try:
                ik = generate_for_objective_strict(
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
                validate_ikizamini(ik)
                elapsed = time.time() - start
                append_job_log(job_id, f"  ✓ Success ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - start
                err = str(e)
                append_job_log(job_id, f"  ✗ FAILED ({elapsed:.1f}s): {obj_id}: {err}")
                failures.append(f"{obj_id}: {err}")

            # Persist outputs immediately (TXT always; JSON only on success)
            txt_path, json_path = write_output_files(job_id, obj_id, obj_text, path, ik, err)

            # Update objective row
            conn = db_connect()
            conn.execute("""
              UPDATE objectives
              SET status=?, finished_at_unix=?, error_text=?, output_txt_path=?, output_json_path=?
              WHERE job_id=? AND objective_id=?
            """, (
                "done" if ik is not None else "failed",
                int(time.time()),
                err,
                txt_path,
                json_path or None,
                job_id,
                obj_id,
            ))
            conn.commit()
            conn.close()

            # Update job completion counts
            conn = db_connect()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS n FROM objectives WHERE job_id=? AND status IN ('done','failed')", (job_id,))
            done_count = cur.fetchone()["n"]
            cur.execute("UPDATE jobs SET completed=?, failures_json=?, heartbeat_unix=? WHERE job_id=?",
                        (done_count, json.dumps(failures, ensure_ascii=False), int(time.time()), job_id))
            conn.commit()
            conn.close()

        # Finish job (even with failures)
        conn = db_connect()
        conn.execute("UPDATE jobs SET status=?, finished_at_unix=?, heartbeat_unix=? WHERE job_id=?",
                     ("done", int(time.time()), int(time.time()), job_id))
        conn.commit()
        conn.close()

        append_job_log(job_id, "============================================================")
        append_job_log(job_id, f"JOB COMPLETED: {job_id}  (outputs persisted to disk + DB)")
        append_job_log(job_id, "============================================================")


def start_job_thread(job_id: str, resume: bool = False) -> None:
    t = threading.Thread(target=run_job, args=(job_id, resume), daemon=True)
    t.start()


# =============================================================================
# UI Templates
# =============================================================================

PAGE_HOME = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>IKIZAMINI (Ollama) UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; }
    .box { max-width: 1100px; }
    .row { display:flex; gap:16px; margin-bottom:12px; flex-wrap: wrap; }
    .row label { width: 180px; font-weight: bold; }
    input[type=text], input[type=number] { width: 420px; padding: 6px; }
    .btn { padding: 10px 14px; background:#111; color:#fff; border:0; cursor:pointer; text-decoration:none; display:inline-block;}
    .btn:hover:not(:disabled) { background:#333; }
    .muted { color:#666; }
    .card { border:1px solid #ddd; padding:16px; margin-top:18px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border:1px solid #eee; padding:8px; text-align:left; }
    th { background:#fafafa; }
    .pill { display:inline-block; padding:4px 10px; border-radius:999px; background:#eee; }
  </style>
</head>
<body>
<div class="box">
  <h1>IKIZAMINI (Local Ollama)</h1>
  <p class="muted">
    Durable mode: outputs are saved to disk + SQLite as the job runs.
    You can recover partial results after a crash.
  </p>

  <div class="card">
    <h2>Start New Job</h2>
    <form method="post" action="/generate" enctype="multipart/form-data">
      <div class="row">
        <label>Objectives .txt</label>
        <input type="file" name="input_file" accept=".txt" required />
      </div>

      <div class="row">
        <label>Ollama URL</label>
        <input type="text" name="ollama_url" value="http://localhost:11434" />
      </div>

      <div class="row">
        <label>Worker model</label>
        <input type="text" name="worker_model" value="qwen:32b" />
      </div>

      <div class="row">
        <label>Manager model</label>
        <input type="text" name="manager_model" value="qwen:32b" />
      </div>

      <div class="row">
        <label>Max rounds</label>
        <input type="number" name="max_rounds" value="6" min="1" max="30" />
      </div>

      <div class="row">
        <label>Context size (num_ctx)</label>
        <input type="number" name="num_ctx" value="8192" min="1024" max="65536" step="256" />
      </div>

      <div class="row">
        <label>Ollama timeout (sec)</label>
        <input type="number" name="timeout_s" value="600" min="60" max="7200" />
      </div>

      <div class="row">
        <label>Retries</label>
        <input type="number" name="max_retries" value="4" min="1" max="20" />
      </div>

      <div class="row">
        <label>Limit objectives</label>
        <input type="number" name="limit" value="0" min="0" max="999999" />
      </div>

      <button class="btn" type="submit">Start Generation</button>
    </form>
  </div>

  <div class="card">
    <h2>Existing Jobs</h2>
    {% if jobs|length == 0 %}
      <p class="muted">(No jobs yet.)</p>
    {% else %}
      <table>
        <thead>
          <tr>
            <th>Job ID</th>
            <th>Status</th>
            <th>Progress</th>
            <th>Created</th>
            <th>Open</th>
            <th>Download ZIP</th>
          </tr>
        </thead>
        <tbody>
        {% for j in jobs %}
          <tr>
            <td><span class="pill">{{ j.job_id }}</span></td>
            <td>{{ j.status }}</td>
            <td>{{ j.completed }}/{{ j.requested }}</td>
            <td>{{ j.created_at_unix }}</td>
            <td><a class="btn" href="/job/{{ j.job_id }}">View</a></td>
            <td><a class="btn" href="/download/zip/{{ j.job_id }}">ZIP (partial ok)</a></td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    {% endif %}
  </div>

</div>
</body>
</html>
"""

PAGE_JOB = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>IKIZAMINI Job {{ job_id }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; }
    .box { max-width: 1100px; }
    .muted { color:#666; }
    .row { display:flex; gap:18px; flex-wrap: wrap; align-items: center; }
    .card { border:1px solid #ddd; padding:16px; margin-top:18px; }
    .btn { padding: 10px 14px; background:#111; color:#fff; border:0; cursor:pointer; text-decoration: none; display:inline-block; }
    .btn:hover:not(.disabled) { background:#333; }
    .btn.disabled { opacity:0.5; cursor:not-allowed; background:#666; pointer-events: none; }
    pre { background:#f7f7f7; padding:12px; overflow:auto; max-height: 420px; }
    .kv { margin: 6px 0; }
    .pill { display:inline-block; padding:4px 10px; border-radius: 999px; background:#eee; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border:1px solid #eee; padding:8px; text-align:left; }
    th { background:#fafafa; }
  </style>
</head>
<body>
<div class="box">
  <h1>IKIZAMINI Job</h1>
  <p class="muted">This page polls the server. Downloads work even while running (partial ZIP/master).</p>

  <div class="row">
    <div class="kv"><b>Job ID:</b> <span class="pill">{{ job_id }}</span></div>
    <div class="kv"><b>Status:</b> <span id="status" class="pill">loading...</span></div>
    <div class="kv"><b>Progress:</b> <span id="progress" class="pill">0/0</span></div>
    <div class="kv"><b>Heartbeat:</b> <span id="heartbeat" class="pill">-</span></div>
  </div>

  <div class="card">
    <div class="row" style="justify-content: space-between;">
      <div>
        <a class="btn" href="/download/master/{{ job_id }}">Download Master .txt (partial ok)</a>
        <a class="btn" href="/download/zip/{{ job_id }}">Download ZIP (partial ok)</a>
      </div>
      <div>
        <a class="btn" href="/">Home</a>
        <a id="resumeBtn" class="btn" href="/resume/{{ job_id }}">Resume (skip done/failed)</a>
      </div>
    </div>
    <p class="muted" style="margin-top:10px;">
      “Resume” is useful after a crash or interruption. It will continue pending objectives.
    </p>
  </div>

  <div class="card">
    <h3>Failures</h3>
    <pre id="failures">(none)</pre>
  </div>

  <div class="card">
    <h3>Objectives</h3>
    <div class="muted">Shows latest 30 objectives by status (server-side).</div>
    <pre id="objectives">Loading...</pre>
  </div>

  <div class="card">
    <h3>Live Log (tail)</h3>
    <pre id="logs">Loading...</pre>
  </div>
</div>

<script>
  const jobId = "{{ job_id }}";

  async function poll() {
    try {
      const res = await fetch(`/api/job/${jobId}`, { cache: "no-store" });
      if (!res.ok) {
        document.getElementById("status").textContent = "error";
        return;
      }
      const data = await res.json();

      document.getElementById("status").textContent = data.status;
      document.getElementById("progress").textContent = `${data.completed}/${data.requested}`;
      document.getElementById("heartbeat").textContent = data.heartbeat_unix || "-";

      const failures = (data.failures && data.failures.length) ? data.failures.join("\\n") : "(none)";
      document.getElementById("failures").textContent = failures;

      document.getElementById("logs").textContent = (data.logs_tail && data.logs_tail.length)
        ? data.logs_tail.join("\\n")
        : "(no logs yet)";

      document.getElementById("objectives").textContent = (data.objectives_preview && data.objectives_preview.length)
        ? data.objectives_preview.join("\\n")
        : "(no objectives yet)";

      // Poll continuously for long runs
      setTimeout(poll, 2000);
    } catch (e) {
      setTimeout(poll, 4000);
    }
  }

  poll();
</script>
</body>
</html>
"""


# =============================================================================
# Routes
# =============================================================================

@app.get("/")
def home():
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT job_id, status, requested, completed, created_at_unix FROM jobs ORDER BY created_at_unix DESC LIMIT 50")
    jobs = cur.fetchall()
    conn.close()
    return render_template_string(PAGE_HOME, jobs=jobs)


@app.get("/job/<job_id>")
def job_view(job_id: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT job_id FROM jobs WHERE job_id=?", (job_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        abort(404)
    return render_template_string(PAGE_JOB, job_id=job_id)


@app.get("/api/job/<job_id>")
def api_job(job_id: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
    job = cur.fetchone()
    if not job:
        conn.close()
        return jsonify({"error": "not_found"}), 404

    failures = json.loads(job["failures_json"] or "[]")

    # Tail of objectives (preview)
    cur.execute("""
      SELECT objective_id, objective_text, status
      FROM objectives
      WHERE job_id=?
      ORDER BY objective_id
      LIMIT 30
    """, (job_id,))
    objs = cur.fetchall()

    conn.close()

    logs_tail = read_log_tail(job["log_path"] or "", max_lines=120)
    obj_preview = [f"{r['objective_id']} - {r['objective_text']} [{r['status']}]" for r in objs]

    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "requested": job["requested"],
        "completed": job["completed"],
        "heartbeat_unix": job["heartbeat_unix"],
        "failures": failures,
        "logs_tail": logs_tail,
        "objectives_preview": obj_preview,
    })


@app.post("/generate")
def generate():
    up = request.files.get("input_file")
    if not up or up.filename == "":
        return "No file uploaded", 400

    content = up.read().decode("utf-8", errors="replace")

    ollama_url = request.form.get("ollama_url", "http://localhost:11434").strip()
    worker_model = request.form.get("worker_model", "qwen:32b").strip()
    manager_model = request.form.get("manager_model", "qwen:32b").strip()
    max_rounds = int(request.form.get("max_rounds", "6"))
    num_ctx = int(request.form.get("num_ctx", "8192"))
    timeout_s = int(request.form.get("timeout_s", str(DEFAULT_TIMEOUT_S)))
    max_retries = int(request.form.get("max_retries", str(DEFAULT_MAX_RETRIES)))
    limit = int(request.form.get("limit", "0"))

    job_id = str(uuid.uuid4())[:8]
    inp_path, log_path = ensure_job_files(job_id)

    # Save input to disk immediately (recoverable)
    with open(inp_path, "w", encoding="utf-8") as f:
        f.write(content)

    config = {
        "ollama_url": ollama_url,
        "worker_model": worker_model,
        "manager_model": manager_model,
        "max_rounds": max_rounds,
        "num_ctx": num_ctx,
        "timeout_s": timeout_s,
        "max_retries": max_retries,
        "limit": limit,
    }

    # Create job in DB
    conn = db_connect()
    conn.execute("""
      INSERT INTO jobs (job_id, status, created_at_unix, heartbeat_unix, input_filename, input_path, config_json, requested, completed, failures_json, master_path, log_path)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        job_id, "queued", int(time.time()), int(time.time()),
        up.filename, inp_path, json.dumps(config, ensure_ascii=False),
        0, 0, "[]", None, log_path
    ))
    conn.commit()
    conn.close()

    append_job_log(job_id, "NEW JOB CREATED (durable mode).")
    append_job_log(job_id, f"Input saved to: {inp_path}")
    append_job_log(job_id, f"Logs saved to: {log_path}")

    # Start background processing
    start_job_thread(job_id, resume=False)

    return redirect(url_for("job_view", job_id=job_id))


@app.get("/resume/<job_id>")
def resume(job_id: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        abort(404)

    # Start resume thread (skip done/failed)
    append_job_log(job_id, "RESUME requested from UI. Will skip done/failed objectives.")
    start_job_thread(job_id, resume=True)

    return redirect(url_for("job_view", job_id=job_id))


@app.get("/download/master/<job_id>")
def download_master(job_id: str):
    # Build partial master at request time (works even mid-run or after crash)
    text = build_master_text(job_id)
    data = text.encode("utf-8", errors="replace")
    return send_file(
        io.BytesIO(data),
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"ikizamini_master_{job_id}.txt",
    )


@app.get("/download/zip/<job_id>")
def download_zip(job_id: str):
    # Always allow partial ZIP
    z = build_zip_from_outputs(job_id)
    return send_file(
        io.BytesIO(z),
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"ikizamini_outputs_{job_id}.zip",
    )


# =============================================================================
# Startup
# =============================================================================

def startup() -> None:
    db_init()
    db_mark_stale_jobs_as_interrupted(stale_after_sec=3600)
    safe_print(f"[{_now_ts()}] IKIZAMINI durable DB: {DB_PATH}")
    safe_print(f"[{_now_ts()}] IKIZAMINI data dir: {DATA_DIR}")


if __name__ == "__main__":
    startup()
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("RUNPOD_PORT", os.environ.get("FLASK_PORT", "8000")))
    safe_print(f"[{_now_ts()}] Starting IKIZAMINI UI on {host}:{port} ...")
    # threaded=True enables concurrent polling while background thread runs
    app.run(host=host, port=port, debug=False, threaded=True)
