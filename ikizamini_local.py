#!/usr/bin/env python3
"""
ikizamini_local.py

IKIZAMINI Local Web UI (Flask) + Ollama with long-running stability + persistence.

Key properties:
- Runs generation in background thread (UI doesn't timeout).
- Persists job state and objective outputs to SQLite + disk as it runs.
- Partial recovery: even if process crashes, already-written outputs remain downloadable.
- Avoids BrokenPipe crashes by logging to files and catching stdout errors.

What this version focuses on:
- Same durable backend (SQLite + on-disk outputs, resumable jobs, partial downloads).
- Modern, centered UI with responsive layout, improved typography, progress bar,
  log viewer, and clear download controls.

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
import math
import threading
import sqlite3
import socket
from difflib import SequenceMatcher
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import (
    Flask,
    request,
    redirect,
    url_for,
    send_file,
    render_template_string,
    abort,
    jsonify,
)
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

# Allow env-driven model defaults (useful on Runpod where you pre-pull a model like gemma3)
DEFAULT_WORKER_MODEL = os.environ.get("IKIZAMINI_DEFAULT_WORKER_MODEL", "qwen:32b")
DEFAULT_MANAGER_MODEL = os.environ.get("IKIZAMINI_DEFAULT_MANAGER_MODEL", DEFAULT_WORKER_MODEL)

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
      cancel_requested INTEGER DEFAULT 0,
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

    # Best-effort migration for existing DBs created before cancel_requested existed
    try:
        conn = db_connect()
        conn.execute("ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER DEFAULT 0;")
        conn.commit()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


def db_mark_stale_jobs_as_interrupted(stale_after_sec: int = 3600) -> None:
    """If server crashed, jobs may be stuck in running/queued. Mark as interrupted if heartbeat is old."""
    now = int(time.time())
    conn = db_connect()
    cur = conn.cursor()

    cur.execute("""
      SELECT job_id, status, heartbeat_unix, created_at_unix
      FROM jobs
      WHERE status IN ('queued','running')
    """)
    rows = cur.fetchall()

    for r in rows:
        hb = r["heartbeat_unix"] or 0
        created = r["created_at_unix"] or now
        if hb and (now - hb) > stale_after_sec:
            cur.execute("UPDATE jobs SET status=? WHERE job_id=?", ("interrupted", r["job_id"]))
        elif (now - created) > stale_after_sec:
            cur.execute("UPDATE jobs SET status=? WHERE job_id=?", ("interrupted", r["job_id"]))

    conn.commit()
    conn.close()


def is_cancel_requested(job_id: str) -> bool:
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("SELECT cancel_requested FROM jobs WHERE job_id=?", (job_id,))
        row = cur.fetchone()
        conn.close()
        return bool(row and int(row["cancel_requested"] or 0) == 1)
    except Exception:
        return False


def request_cancel(job_id: str) -> None:
    conn = db_connect()
    conn.execute("UPDATE jobs SET cancel_requested=1 WHERE job_id=?", (job_id,))
    conn.commit()
    conn.close()


def mark_remaining_objectives_cancelled(job_id: str) -> None:
    """Mark any objectives not completed as cancelled."""
    conn = db_connect()
    conn.execute(
        """
        UPDATE objectives
        SET status='cancelled', finished_at_unix=?, error_text=COALESCE(error_text,'Cancelled by user')
        WHERE job_id=? AND status IN ('pending')
        """,
        (int(time.time()), job_id),
    )
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
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

    safe_print(line)


def read_log_tail(log_path: str, max_lines: int = 200) -> List[str]:
    """Read last N lines from a potentially large file efficiently."""
    if not log_path or not os.path.exists(log_path):
        return []
    try:
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
    return id_depth(obj_id) >= 4  # 1.1.1.1+


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

    timeout = (10, timeout_s)  # (connect, read)
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

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "being", "been", "as", "at", "from",
    "find", "solve", "compute", "determine", "evaluate", "simplify", "factor",
    "given", "let", "if", "then", "which", "what",
}


def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(s: str) -> set[str]:
    s = _normalize_text(s)
    toks = re.findall(r"[a-z0-9]+", s)
    return {t for t in toks if len(t) >= 3 and t not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()


def _question_signature(q: Dict[str, Any]) -> str:
    stem = str(q.get("stem", "")).strip()
    ch = q.get("choices", {}) if isinstance(q.get("choices"), dict) else {}
    choices = " | ".join([str(ch.get(k, "")).strip() for k in ["A", "B", "C", "D"]])
    return f"{stem}\n{choices}"


def _pairwise_distinctness_issues(qs: List[Dict[str, Any]]) -> List[str]:
    """Flag near-duplicate questions using a 3σ rule + hard ceiling.

    We compute pairwise similarity on stem and (stem+choices). If any pair is an
    unusually high-similarity outlier (>= mean + 3*std) OR exceeds a hard ceiling,
    we treat it as "not distinct" and force repair.
    """
    issues: List[str] = []
    if len(qs) < 2:
        return issues

    pairs: List[tuple[int, int, float]] = []
    scores: List[float] = []

    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            qi = qs[i]
            qj = qs[j]

            sig_i = _question_signature(qi)
            sig_j = _question_signature(qj)

            sim_stem = max(
                _seq_ratio(qi.get("stem", ""), qj.get("stem", "")),
                _jaccard(_tokenize(qi.get("stem", "")), _tokenize(qj.get("stem", ""))),
            )
            sim_sig = max(
                _seq_ratio(sig_i, sig_j),
                _jaccard(_tokenize(sig_i), _tokenize(sig_j)),
            )

            sim = max(sim_stem, sim_sig)
            pairs.append((i, j, sim))
            scores.append(sim)

    if not scores:
        return issues

    mean = sum(scores) / len(scores)
    var = sum((x - mean) ** 2 for x in scores) / len(scores)
    std = math.sqrt(var)

    hard_ceiling = 0.80
    sigma_ceiling = mean + 3.0 * std
    threshold = max(hard_ceiling, sigma_ceiling)

    for i, j, sim in pairs:
        if sim >= threshold:
            issues.append(
                f"Q{i+1} and Q{j+1} are too similar (similarity={sim:.2f}, threshold={threshold:.2f}). "
                "Regenerate at least one so they test a different sub-skill/representation."
            )

    return issues


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

    # Near-duplicate detection (3σ distinctness rule)
    dict_qs = [q for q in qs if isinstance(q, dict)]
    issues.extend(_pairwise_distinctness_issues(dict_qs))
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

Distinctness requirement (3σ rule):
- No two questions may be "near-duplicates". Treat two questions as duplicates if they are very similar in
  wording/structure OR only differ by numbers. Each question must assess a different sub-skill or representation
  (e.g., factoring by GCF vs factoring trinomials vs factor theorem vs synthetic division, etc.).

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


def write_master_incremental(roots: List[ObjNode], outputs_by_id: Dict[str, str], meta: Dict[str, Any]) -> str:
    def heading_line(node: ObjNode) -> str:
        return f"{node.obj_id} {node.title}"

    def render(node: ObjNode, level: int) -> List[str]:
        indent = "   " * max(level - 1, 0)
        out: List[str] = [f"{indent}{heading_line(node)}"]

        if is_learning_objective_id(node.obj_id):
            block = outputs_by_id.get(node.obj_id)
            if block:
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
    jd = job_dir(job_id)
    inp = os.path.join(jd, "input.txt")
    logp = os.path.join(jd, "job.log")
    if not os.path.exists(logp):
        with open(logp, "a", encoding="utf-8") as f:
            f.write(f"[{_now_ts()}] Log created for job {job_id}\n")
    return inp, logp


def write_output_files(job_id: str, obj_id: str, obj_text: str, path: List[Tuple[str, str]],
                       ik_set: Optional[Dict[str, Any]], error: Optional[str]) -> Tuple[str, str]:
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

    return txt_path, ""


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
    return write_master_incremental(roots, outputs_by_id, meta)


def build_zip_from_outputs(job_id: str) -> bytes:
    od = outputs_dir(job_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(od):
            for fn in files:
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, job_dir(job_id))
                zf.write(full, arcname=arc)

        zf.writestr("outputs/index.txt", build_index_text(job_id))
        zf.writestr("master_partial.txt", build_master_text(job_id))

    return buf.getvalue()


# =============================================================================
# Job runner (background)
# =============================================================================

def update_job_heartbeat(job_id: str) -> None:
    conn = db_connect()
    conn.execute("UPDATE jobs SET heartbeat_unix=? WHERE job_id=?", (int(time.time()), job_id))
    conn.commit()
    conn.close()


def run_job(job_id: str, resume: bool = False) -> None:
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

        conn = db_connect()
        conn.execute("UPDATE jobs SET requested=? WHERE job_id=?", (len(objectives), job_id))
        conn.commit()
        conn.close()

        failures: List[str] = json.loads(job["failures_json"] or "[]")
        last_hb = time.time()

        for idx, obj in enumerate(objectives, start=1):
            if is_cancel_requested(job_id):
                append_job_log(job_id, "CANCEL REQUESTED: stopping job (sequential).")
                conn = db_connect()
                conn.execute("UPDATE jobs SET status=?, heartbeat_unix=? WHERE job_id=?",
                             ("cancelling", int(time.time()), job_id))
                conn.commit()
                conn.close()
                break

            if time.time() - last_hb >= HEARTBEAT_EVERY_SEC:
                update_job_heartbeat(job_id)
                last_hb = time.time()

            obj_id = obj["objective_id"]
            obj_text = obj["objective_text"]
            path = obj["path"]

            if resume:
                conn = db_connect()
                cur = conn.cursor()
                cur.execute("SELECT status FROM objectives WHERE job_id=? AND objective_id=?", (job_id, obj_id))
                row = cur.fetchone()
                conn.close()
                if row and row["status"] in ("done", "failed"):
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

            txt_path, json_path = write_output_files(job_id, obj_id, obj_text, path, ik, err)

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

            conn = db_connect()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) AS n FROM objectives WHERE job_id=? AND status IN ('done','failed')", (job_id,))
            done_count = cur.fetchone()["n"]
            cur.execute("UPDATE jobs SET completed=?, failures_json=?, heartbeat_unix=? WHERE job_id=?",
                        (done_count, json.dumps(failures, ensure_ascii=False), int(time.time()), job_id))
            conn.commit()
            conn.close()

        if is_cancel_requested(job_id):
            mark_remaining_objectives_cancelled(job_id)
            conn = db_connect()
            conn.execute("UPDATE jobs SET status=?, finished_at_unix=?, heartbeat_unix=? WHERE job_id=?",
                         ("cancelled", int(time.time()), int(time.time()), job_id))
            conn.commit()
            conn.close()
        else:
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
# Modern centered UI templates
# =============================================================================

UI_BASE_CSS = r"""
:root{
  --bg0:#0b1020;
  --bg1:#0e1630;
  --card:rgba(255,255,255,.06);
  --card2:rgba(255,255,255,.08);
  --stroke:rgba(255,255,255,.10);
  --stroke2:rgba(255,255,255,.16);
  --text:rgba(255,255,255,.92);
  --muted:rgba(255,255,255,.68);
  --faint:rgba(255,255,255,.50);
  --brand:#7c5cff;
  --brand2:#37d1ff;
  --good:#37d67a;
  --warn:#ffcc66;
  --bad:#ff5c77;
  --shadow: 0 18px 60px rgba(0,0,0,.45);
  --radius: 18px;
  --radius2: 12px;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --sans: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, "Apple Color Emoji","Segoe UI Emoji";
}

*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0;
  font-family:var(--sans);
  color:var(--text);
  background:
    radial-gradient(1200px 800px at 10% 10%, rgba(124,92,255,.20), transparent 55%),
    radial-gradient(1000px 700px at 90% 20%, rgba(55,209,255,.18), transparent 55%),
    radial-gradient(900px 650px at 40% 100%, rgba(55,214,122,.10), transparent 55%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
}

a{color:inherit}
.container{
  min-height:100%;
  display:flex;
  align-items:center;
  justify-content:center;
  padding:28px 18px;
}

.shell{
  width:min(1100px, 100%);
}

.topbar{
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap:16px;
  margin-bottom:18px;
}

.brand{
  display:flex;
  flex-direction:column;
  gap:6px;
}
.brand h1{
  margin:0;
  letter-spacing:.2px;
  font-size: 28px;
  line-height:1.12;
}
.brand .sub{
  color:var(--muted);
  font-size:14px;
  max-width: 72ch;
}

.pills{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  justify-content:flex-end;
}
.pill{
  border:1px solid var(--stroke);
  background: rgba(255,255,255,.04);
  padding:8px 10px;
  border-radius:999px;
  font-size:12px;
  color:var(--muted);
  backdrop-filter: blur(8px);
}

.grid{
  display:grid;
  grid-template-columns: 1.15fr .85fr;
  gap:16px;
}
@media (max-width: 980px){
  .grid{grid-template-columns: 1fr;}
  .pills{justify-content:flex-start;}
}

.card{
  border:1px solid var(--stroke);
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  overflow:hidden;
}

.card header{
  padding:14px 16px;
  border-bottom:1px solid var(--stroke);
  background: rgba(255,255,255,.03);
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
}
.hdrLeft{display:flex; flex-direction:column; gap:4px; min-width:0;}
.hdrRight{display:flex; align-items:center; gap:10px;}

.collapseBtn{
  appearance:none;
  border:1px solid rgba(255,255,255,.14);
  background: rgba(255,255,255,.05);
  color: rgba(255,255,255,.86);
  border-radius: 10px;
  width: 34px;
  height: 34px;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  cursor:pointer;
  transition: transform .18s ease, background .18s ease, border-color .18s ease;
  flex: 0 0 auto;
}
.collapseBtn:hover{ background: rgba(255,255,255,.09); border-color: rgba(255,255,255,.22); }
.card.collapsed .body{ display:none; }
.card.collapsed .collapseBtn{ transform: rotate(-90deg); }
.card header h2{
  margin:0;
  font-size:14px;
  letter-spacing:.2px;
  color: rgba(255,255,255,.86);
}
.card header .hint{
  color:var(--muted);
  font-size:12px;
}

.card .body{
  padding:16px;
}

.formgrid{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap:12px;
}
@media (max-width: 700px){
  .formgrid{grid-template-columns: 1fr;}
}

.field{
  display:flex;
  flex-direction:column;
  gap:7px;
}
label{
  font-size:12px;
  color: var(--muted);
}
input[type=text], input[type=number], input[type=file]{
  width:100%;
  padding:11px 12px;
  border-radius: 12px;
  border:1px solid var(--stroke2);
  background: rgba(0,0,0,.24);
  color: var(--text);
  outline:none;
}
input[type=text]:focus, input[type=number]:focus, input[type=file]:focus{
  border-color: rgba(124,92,255,.65);
  box-shadow: 0 0 0 4px rgba(124,92,255,.18);
}

.actions{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  margin-top:14px;
}

.btn{
  appearance:none;
  border:1px solid var(--stroke2);
  background: rgba(255,255,255,.06);
  color: var(--text);
  padding:10px 12px;
  border-radius: 12px;
  cursor:pointer;
  font-weight:600;
  font-size:13px;
  text-decoration:none;
  display:inline-flex;
  align-items:center;
  gap:10px;
  transition: transform .08s ease, background .2s ease, border-color .2s ease;
}
.btn:hover{ background: rgba(255,255,255,.10); }
.btn:active{ transform: translateY(1px); }
.btn.primary{
  border-color: rgba(124,92,255,.55);
  background: linear-gradient(135deg, rgba(124,92,255,.38), rgba(55,209,255,.22));
}
.btn.primary:hover{ background: linear-gradient(135deg, rgba(124,92,255,.48), rgba(55,209,255,.30)); }
.btn.good{
  border-color: rgba(55,214,122,.55);
  background: rgba(55,214,122,.10);
}
.btn.bad{
  border-color: rgba(255,92,119,.55);
  background: rgba(255,92,119,.10);
}

.small{
  font-size:12px;
  color: var(--muted);
}

.jobs{
  width:100%;
  border-collapse: collapse;
}
.jobs th, .jobs td{
  padding:10px 10px;
  border-bottom:1px solid rgba(255,255,255,.07);
  font-size:13px;
}
.jobs th{color: var(--muted); font-weight:600; text-align:left;}
.jobs td{color: rgba(255,255,255,.86);}
.tag{
  font-family: var(--mono);
  font-size:12px;
  border:1px solid rgba(255,255,255,.14);
  background: rgba(0,0,0,.22);
  padding:4px 8px;
  border-radius:999px;
  display:inline-block;
}
.status{
  font-size:12px;
  padding:4px 8px;
  border-radius:999px;
  display:inline-block;
  border:1px solid rgba(255,255,255,.12);
}
.status.running{ color: rgba(55,209,255,.95); border-color: rgba(55,209,255,.35); background: rgba(55,209,255,.08); }
.status.done{ color: rgba(55,214,122,.95); border-color: rgba(55,214,122,.35); background: rgba(55,214,122,.08); }
.status.error{ color: rgba(255,92,119,.95); border-color: rgba(255,92,119,.35); background: rgba(255,92,119,.08); }
.status.queued{ color: rgba(255,204,102,.95); border-color: rgba(255,204,102,.35); background: rgba(255,204,102,.08); }
.status.interrupted{ color: rgba(255,204,102,.95); border-color: rgba(255,204,102,.35); background: rgba(255,204,102,.08); }
.status.cancelling{ color: rgba(255,204,102,.95); border-color: rgba(255,204,102,.35); background: rgba(255,204,102,.08); }
.status.cancelled{ color: rgba(255,92,119,.95); border-color: rgba(255,92,119,.35); background: rgba(255,92,119,.08); }

.hr{height:1px; background: rgba(255,255,255,.08); margin:12px 0;}

.kv{
  display:flex;
  gap:10px;
  flex-wrap:wrap;
  align-items:center;
}
.kv .k{color: var(--muted); font-size:12px;}
.kv .v{font-family: var(--mono); font-size:12px; color: rgba(255,255,255,.86); padding:4px 8px; border:1px solid rgba(255,255,255,.12); border-radius:10px; background: rgba(0,0,0,.18);}

.progressWrap{
  width:100%;
  height:10px;
  border-radius:999px;
  border: 1px solid rgba(255,255,255,.12);
  background: rgba(0,0,0,.22);
  overflow:hidden;
}
.progressBar{
  height:100%;
  width:0%;
  background: linear-gradient(90deg, rgba(124,92,255,.9), rgba(55,209,255,.8));
}

pre{
  margin:0;
  font-family: var(--mono);
  font-size:12px;
  line-height: 1.4;
  color: rgba(255,255,255,.84);
  background: rgba(0,0,0,.28);
  border:1px solid rgba(255,255,255,.10);
  border-radius: 14px;
  padding:12px;
  max-height: 420px;
  overflow:auto;
}
"""

PAGE_HOME = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{{ ui_title }} · Durable Ollama UI</title>
  <style>{{ ui_css }}</style>
</head>
<body>
<div class="container">
  <div class="shell">
    <div class="topbar">
      <div class="brand">
        <h1>{{ ui_title }}</h1>
        <div class="sub">
          Durable Ollama generator. Outputs are written to disk and indexed in SQLite as the job runs,
          so you can recover and download partial results even after a crash.
        </div>
      </div>
      <div class="pills">
        <div class="pill">DB: <span style="font-family:var(--mono)">{{ db_path }}</span></div>
        <div class="pill">Data: <span style="font-family:var(--mono)">{{ data_dir }}</span></div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <header>
          <div class="hdrLeft">
            <h2>Start a new job</h2>
            <div class="hint">Runs in the background. UI stays responsive.</div>
          </div>
          <div class="hdrRight">
            <button class="collapseBtn" type="button" data-collapse aria-label="Collapse/expand">▾</button>
          </div>
        </header>
        <div class="body">
          <form method="post" action="/generate" enctype="multipart/form-data">
            <div class="formgrid">
              <div class="field">
                <label>Objectives (.txt)</label>
                <input type="file" name="input_file" accept=".txt" required />
              </div>

              <div class="field">
                <label>Ollama URL</label>
                <input type="text" name="ollama_url" value="http://localhost:11434" />
              </div>

              <div class="field">
                <label>Worker model</label>
                <input type="text" name="worker_model" value="{{ default_worker_model }}" />
              </div>

              <div class="field">
                <label>Manager model</label>
                <input type="text" name="manager_model" value="{{ default_manager_model }}" />
              </div>

              <div class="field">
                <label>Max rounds</label>
                <input type="number" name="max_rounds" value="6" min="1" max="30" />
              </div>

              <div class="field">
                <label>Context size (num_ctx)</label>
                <input type="number" name="num_ctx" value="8192" min="1024" max="65536" step="256" />
              </div>

              <div class="field">
                <label>Ollama timeout (sec)</label>
                <input type="number" name="timeout_s" value="600" min="60" max="7200" />
              </div>

              <div class="field">
                <label>Retries</label>
                <input type="number" name="max_retries" value="4" min="1" max="20" />
              </div>

              <div class="field">
                <label>Limit objectives (0 = no limit)</label>
                <input type="number" name="limit" value="0" min="0" max="999999" />
              </div>

              <div class="field">
                <label>Notes</label>
                <input type="text" value="Downloads remain available while running (partial ok)." readonly />
              </div>
            </div>

            <div class="actions">
              <button class="btn primary" type="submit">Start generation</button>
              <span class="small">Tip: For very large jobs, increase timeout to 900–1800s.</span>
            </div>
          </form>
        </div>
      </div>

      <div class="card">
        <header>
          <div class="hdrLeft">
            <h2>Recent jobs</h2>
            <div class="hint">Open a job to view live logs and downloads.</div>
          </div>
          <div class="hdrRight">
            <button class="collapseBtn" type="button" data-collapse aria-label="Collapse/expand">▾</button>
          </div>
        </header>
        <div class="body">
          {% if jobs|length == 0 %}
            <div class="small">(No jobs yet.)</div>
          {% else %}
            <table class="jobs">
              <thead>
                <tr>
                  <th>Job</th>
                  <th>Status</th>
                  <th>Progress</th>
                  <th>Open</th>
                  <th>ZIP</th>
                </tr>
              </thead>
              <tbody>
              {% for j in jobs %}
                <tr>
                  <td><span class="tag">{{ j.job_id }}</span></td>
                  <td><span class="status {{ j.status }}">{{ j.status }}</span></td>
                  <td>{{ j.completed }}/{{ j.requested }}</td>
                  <td><a class="btn" href="/job/{{ j.job_id }}">View</a></td>
                  <td><a class="btn" href="/download/zip/{{ j.job_id }}">Download</a></td>
                </tr>
              {% endfor %}
              </tbody>
            </table>
          {% endif %}
          <div class="hr"></div>
          <div class="small">
            Files are stored under <span style="font-family:var(--mono)">{{ abs_data_dir }}</span>.
          </div>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
  // Collapsible panels
  (function () {
    const btns = document.querySelectorAll("[data-collapse]");
    btns.forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        const card = btn.closest(".card");
        if (!card) return;
        card.classList.toggle("collapsed");
      });
    });
  })();
</script>
</body>
</html>
"""


PAGE_JOB = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{{ ui_title }} · Job {{ job_id }}</title>
  <style>{{ ui_css }}</style>
</head>
<body>
<div class="container">
  <div class="shell">
    <div class="topbar">
      <div class="brand">
        <h1>{{ ui_title }} · <span style="font-family:var(--mono); font-weight:700;">{{ job_id }}</span></h1>
        <div class="sub">
          Live status updates, log tail, and downloads. Partial downloads work during execution.
        </div>
      </div>
      <div class="pills">
        <div class="pill">DB: <span style="font-family:var(--mono)">{{ db_path }}</span></div>
        <div class="pill">Data: <span style="font-family:var(--mono)">{{ data_dir }}</span></div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <header>
          <div class="hdrLeft">
            <h2>Status</h2>
            <div class="hint">Polls every 2 seconds.</div>
          </div>
          <div class="hdrRight">
            <button class="collapseBtn" type="button" data-collapse aria-label="Collapse/expand">▾</button>
          </div>
        </header>
        <div class="body">
          <div class="kv" style="margin-bottom:10px;">
            <div class="k">Status</div><div class="v" id="status">loading</div>
            <div class="k">Progress</div><div class="v" id="progress">0/0</div>
            <div class="k">Heartbeat</div><div class="v" id="heartbeat">-</div>
          </div>

          <div class="progressWrap" aria-label="progress">
            <div class="progressBar" id="pbar"></div>
          </div>

          <div class="actions" style="margin-top:14px;">
            <a class="btn primary" href="/download/master/{{ job_id }}">Download master (.txt)</a>
            <a class="btn" href="/download/zip/{{ job_id }}">Download ZIP (partial ok)</a>
            <a class="btn good" href="/resume/{{ job_id }}" title="Resume skips done/failed objectives">Resume</a>
            <form method="post" action="/terminate/{{ job_id }}" style="display:inline;">
              <button class="btn" type="submit"
                style="border-color: rgba(255,92,119,.35); background: rgba(255,92,119,.10);"
                onclick="return confirm('Terminate this job? In-flight requests may finish, but no new objectives will start.');">
                Terminate
              </button>
            </form>
            <a class="btn" href="/">Home</a>
          </div>

          <div class="hr"></div>
          <div class="small">
            If the server restarts, jobs in running/queued with a stale heartbeat become <b>interrupted</b>.
            You can press <b>Resume</b> to continue pending objectives.
          </div>
        </div>
      </div>

      <div class="card">
        <header>
          <div class="hdrLeft">
            <h2>Failures</h2>
            <div class="hint">Aggregated from DB.</div>
          </div>
          <div class="hdrRight">
            <button class="collapseBtn" type="button" data-collapse aria-label="Collapse/expand">▾</button>
          </div>
        </header>
        <div class="body">
          <pre id="failures">(none)</pre>
        </div>
      </div>

      <div class="card" style="grid-column: 1 / -1;">
        <header>
          <div class="hdrLeft">
            <h2>Objectives (preview)</h2>
            <div class="hint">First 30 objectives ordered by ID.</div>
          </div>
          <div class="hdrRight">
            <button class="collapseBtn" type="button" data-collapse aria-label="Collapse/expand">▾</button>
          </div>
        </header>
        <div class="body">
          <pre id="objectives">Loading…</pre>
        </div>
      </div>

      <div class="card" style="grid-column: 1 / -1;">
        <header>
          <div class="hdrLeft">
            <h2>Live log (tail)</h2>
            <div class="hint">Last ~200 lines from job.log.</div>
          </div>
          <div class="hdrRight">
            <button class="collapseBtn" type="button" data-collapse aria-label="Collapse/expand">▾</button>
          </div>
        </header>
        <div class="body">
          <pre id="logs">Loading…</pre>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
  // Collapsible panels
  (function () {
    const btns = document.querySelectorAll("[data-collapse]");
    btns.forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        const card = btn.closest(".card");
        if (!card) return;
        card.classList.toggle("collapsed");
      });
    });
  })();

  const jobId = "{{ job_id }}";

  async function poll() {
    try {
      const res = await fetch(`/api/job/${jobId}`, { cache: "no-store" });
      if (!res.ok) return setTimeout(poll, 4000);
      const data = await res.json();

      document.getElementById("status").textContent = data.status;
      document.getElementById("progress").textContent = `${data.completed}/${data.requested}`;
      document.getElementById("heartbeat").textContent = data.heartbeat_unix || "-";

      const pct = (data.requested > 0) ? Math.min(100, Math.floor((data.completed / data.requested) * 100)) : 0;
      document.getElementById("pbar").style.width = pct + "%";

      document.getElementById("failures").textContent =
        (data.failures && data.failures.length) ? data.failures.join("\\n") : "(none)";

      document.getElementById("logs").textContent =
        (data.logs_tail && data.logs_tail.length) ? data.logs_tail.join("\\n") : "(no logs yet)";

      document.getElementById("objectives").textContent =
        (data.objectives_preview && data.objectives_preview.length) ? data.objectives_preview.join("\\n") : "(no objectives yet)";

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

    return render_template_string(
        PAGE_HOME,
        jobs=jobs,
        ui_css=UI_BASE_CSS,
        ui_title=os.environ.get("IKIZAMINI_UI_TITLE", "IKIZAMINI"),
        default_worker_model=DEFAULT_WORKER_MODEL,
        default_manager_model=DEFAULT_MANAGER_MODEL,
        db_path=DB_PATH,
        data_dir=DATA_DIR,
        abs_data_dir=os.path.abspath(DATA_DIR),
    )


@app.get("/job/<job_id>")
def job_view(job_id: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT job_id FROM jobs WHERE job_id=?", (job_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        abort(404)

    return render_template_string(
        PAGE_JOB,
        job_id=job_id,
        ui_css=UI_BASE_CSS,
        ui_title=os.environ.get("IKIZAMINI_UI_TITLE", "IKIZAMINI"),
        db_path=DB_PATH,
        data_dir=DATA_DIR,
    )


@app.post("/terminate/<job_id>")
def terminate_job(job_id: str):
    conn = db_connect()
    cur = conn.cursor()
    cur.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        abort(404)

    status = row["status"]
    if status in ("done", "error", "cancelled"):
        return redirect(f"/job/{job_id}")

    request_cancel(job_id)
    append_job_log(job_id, "Terminate requested from UI.")
    conn = db_connect()
    conn.execute("UPDATE jobs SET status=?, heartbeat_unix=? WHERE job_id=?",
                 ("cancelling", int(time.time()), job_id))
    conn.commit()
    conn.close()
    return redirect(f"/job/{job_id}")



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

    cur.execute("""
      SELECT objective_id, objective_text, status
      FROM objectives
      WHERE job_id=?
      ORDER BY objective_id
      LIMIT 30
    """, (job_id,))
    objs = cur.fetchall()
    conn.close()

    logs_tail = read_log_tail(job["log_path"] or "", max_lines=200)
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
    worker_model = request.form.get("worker_model", DEFAULT_WORKER_MODEL).strip()
    manager_model = request.form.get("manager_model", DEFAULT_MANAGER_MODEL).strip()
    max_rounds = int(request.form.get("max_rounds", "6"))
    num_ctx = int(request.form.get("num_ctx", "8192"))
    timeout_s = int(request.form.get("timeout_s", str(DEFAULT_TIMEOUT_S)))
    max_retries = int(request.form.get("max_retries", str(DEFAULT_MAX_RETRIES)))
    limit = int(request.form.get("limit", "0"))

    job_id = str(uuid.uuid4())[:8]
    inp_path, log_path = ensure_job_files(job_id)

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

    append_job_log(job_id, "RESUME requested from UI. Will skip done/failed objectives.")
    start_job_thread(job_id, resume=True)
    return redirect(url_for("job_view", job_id=job_id))


@app.get("/download/master/<job_id>")
def download_master(job_id: str):
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


def _is_port_free(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        return True
    except OSError:
        return False


def _pick_free_port(host: str, preferred_ports: List[int]) -> int:
    for p in preferred_ports:
        if _is_port_free(host, p):
            return p
    raise RuntimeError(f"No free port found among: {preferred_ports}")


if __name__ == "__main__":
    os.environ.setdefault("IKIZAMINI_UI_TITLE", "Ikizamini Sequential Processor")
    startup()
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port_env = os.environ.get("RUNPOD_PORT") or os.environ.get("FLASK_PORT")
    if port_env:
        port = int(port_env)
    else:
        # Avoid collisions on shared environments (e.g., port 8000 in use).
        port = _pick_free_port(host, [8000, 8001, 5000, 5001, 8080, 8081])
    safe_print(f"[{_now_ts()}] Starting IKIZAMINI UI on {host}:{port} ...")
    app.run(host=host, port=port, debug=False, threaded=True)
