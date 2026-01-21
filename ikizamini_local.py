#!/usr/bin/env python3
"""
ikizamini_local.py

Local Web UI (Flask) for IKIZAMINI using Ollama.

Features:
- Upload objectives .txt
- Configure Ollama URL, worker model, manager model, max rounds, context size
- Generate master .txt + per-objective .txt files
- Download master .txt or ZIP of per-objective outputs

Run:
  python3 ikizamini_ui.py
Then open:
  http://127.0.0.1:5000

Notes:
- Requires Ollama running at http://localhost:11434 by default
- Pull models first (example):
    ollama pull llama3.1:8b
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
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, request, redirect, url_for, send_file, render_template_string, abort
from jsonschema import validate as jsonschema_validate, ValidationError

# Logging helper
def log_progress(message: str):
    """Print progress message with timestamp to terminal"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)
    sys.stdout.flush()


# =============================================================================
# Core generator (Ollama)
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

def backoff_sleep(attempt: int) -> None:
    import time as _t
    base = min(2 ** attempt, 10)
    jitter = random.uniform(0.0, 0.4)
    _t.sleep(base + jitter)

def ollama_chat(base_url: str, model: str, system: str, user: str,
                temperature: float, num_ctx: int, timeout_s: int) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data.get("message", {}) or {}).get("content", "").strip()

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
JSON_OBJECT_RE = re.compile(r"(\{.*\})", re.DOTALL)

def extract_json_object(text: str) -> Dict[str, Any]:
    m = JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))
    m2 = JSON_OBJECT_RE.search(text)
    if m2:
        return json.loads(m2.group(1))
    raise ValueError("No JSON object found in model output.")

def call_ollama_json(base_url: str, model: str, system: str, user: str,
                     validate_fn, temperature: float = 0.2, num_ctx: int = 8192,
                     timeout_s: int = 180, max_retries: int = 4) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            raw = ollama_chat(base_url, model, system, user, temperature, num_ctx, timeout_s)
            obj = extract_json_object(raw)
            validate_fn(obj)
            return obj
        except Exception as e:
            last_err = e
            backoff_sleep(attempt)
    raise RuntimeError(f"Ollama call failed after retries. Last error: {last_err}")

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
                "type": "object", "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "stem": {"type": "string"},
                    "choices": {
                        "type": "object", "additionalProperties": False,
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
Return ONLY JSON matching the manager schema. No commentary.
"""

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

def worker_generate(base_url: str, model: str, objective_id: str, objective_text: str, num_ctx: int) -> Dict[str, Any]:
    user = f"""Learning objective:
- id: {objective_id}
- text: {objective_text}

Generate 7 exam questions that directly assess this objective.
Return ONLY a JSON object (no markdown)."""
    return call_ollama_json(base_url, model, WORKER_SYSTEM, user, validate_ikizamini, temperature=0.25, num_ctx=num_ctx)

def worker_repair(base_url: str, model: str, objective_id: str, objective_text: str,
                  candidate: Dict[str, Any], issues: List[str], num_ctx: int) -> Dict[str, Any]:
    user = f"""Learning objective:
- id: {objective_id}
- text: {objective_text}

Issues to fix:
{json.dumps(issues, ensure_ascii=False, indent=2)}

Current JSON:
{json.dumps(candidate, ensure_ascii=False)}

Return ONLY a corrected JSON object (no markdown)."""
    return call_ollama_json(base_url, model, WORKER_REPAIR_SYSTEM, user, validate_ikizamini, temperature=0.2, num_ctx=num_ctx)

def manager_review(base_url: str, model: str, objective_id: str, objective_text: str,
                   candidate: Dict[str, Any], num_ctx: int) -> Dict[str, Any]:
    user = f"""Learning objective:
- id: {objective_id}
- text: {objective_text}

Candidate JSON:
{json.dumps(candidate, ensure_ascii=False)}

Return ONLY a JSON object matching the manager schema (no markdown)."""
    return call_ollama_json(base_url, model, MANAGER_SYSTEM, user, validate_manager, temperature=0.1, num_ctx=num_ctx)

def generate_for_objective_strict(base_url: str, worker_model: str, manager_model: str,
                                  objective_id: str, objective_text: str,
                                  max_rounds: int, num_ctx: int) -> Dict[str, Any]:
    candidate = worker_generate(base_url, worker_model, objective_id, objective_text, num_ctx=num_ctx)

    for _ in range(max_rounds):
        issues = local_issues(candidate)
        if issues:
            candidate = worker_repair(base_url, worker_model, objective_id, objective_text, candidate, issues, num_ctx=num_ctx)
            continue

        review = manager_review(base_url, manager_model, objective_id, objective_text, candidate, num_ctx=num_ctx)

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
        candidate = worker_repair(base_url, worker_model, objective_id, objective_text, candidate, mgr_issues, num_ctx=num_ctx)

    raise RuntimeError(f"Could not reach manager approval after {max_rounds} rounds for objective {objective_id}.")


# =============================================================================
# Output formatting helpers
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

def write_master_from_tree(roots: List[ObjNode], generated_by_id: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> str:
    def heading_line(node: ObjNode) -> str:
        return f"{node.obj_id} {node.title}"

    def render(node: ObjNode, level: int) -> List[str]:
        indent = "   " * max(level - 1, 0)
        out: List[str] = [f"{indent}{heading_line(node)}"]

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

    lines: List[str] = []
    lines.append("IKIZAMINI MASTER OUTPUT (Ollama UI)")
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
        lines.append("\nSuggested action: increase max rounds or switch models.\n")

    return "\n".join(lines).rstrip() + "\n"


# =============================================================================
# Flask UI
# =============================================================================

app = Flask(__name__)

# In-memory job store. For a single-user local app this is fine.
JOBS: Dict[str, Dict[str, Any]] = {}

PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>IKIZAMINI (Ollama) UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; }
    .box { max-width: 980px; }
    .row { display:flex; gap:16px; margin-bottom:12px; }
    .row label { width: 180px; font-weight: bold; }
    input[type=text], input[type=number] { width: 420px; padding: 6px; }
    .btn { padding: 10px 14px; background:#111; color:#fff; border:0; cursor:pointer; transition: all 0.3s; }
    .btn:hover:not(:disabled) { background:#333; }
    .btn:disabled { opacity:0.5; cursor:not-allowed; background:#666; }
    .btn.loading { background:#0066cc; position:relative; }
    .btn.loading::after { content: '...'; animation: dots 1.5s steps(4, end) infinite; }
    @keyframes dots { 0%, 20% { content: '.'; } 40% { content: '..'; } 60%, 100% { content: '...'; } }
    .card { border:1px solid #ddd; padding:16px; margin-top:18px; }
    .muted { color:#666; }
    pre { background:#f7f7f7; padding:12px; overflow:auto; }
    .links a { margin-right:12px; }
    .status-message { margin-top: 12px; padding: 10px; background: #e8f4f8; border-left: 4px solid #0066cc; display: none; }
    .status-message.show { display: block; }
  </style>
  <script>
    document.addEventListener('DOMContentLoaded', function() {
      const form = document.querySelector('form');
      const submitBtn = document.querySelector('button[type="submit"]');
      const originalBtnText = submitBtn.textContent;
      
      // Add status message div
      const statusMsg = document.createElement('div');
      statusMsg.className = 'status-message';
      statusMsg.textContent = 'Processing request... Please wait.';
      form.appendChild(statusMsg);
      
      form.addEventListener('submit', function(e) {
        // Show visual feedback immediately when button is clicked
        submitBtn.disabled = true;
        submitBtn.classList.add('loading');
        submitBtn.textContent = 'Generating...';
        statusMsg.classList.add('show');
        statusMsg.textContent = 'Request submitted! Processing objectives... This may take a while. Please check the terminal for progress.';
        
        // Allow form to submit normally (don't prevent default)
        // The button will stay disabled until page reloads
      });
    });
  </script>
</head>
<body>
<div class="box">
  <h1>IKIZAMINI (Local Ollama)</h1>
  <p class="muted">Upload objectives .txt, generate master output and a ZIP of per-objective files.</p>

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
      <input type="number" name="max_rounds" value="6" min="1" max="20" />
    </div>

    <div class="row">
      <label>Context size (num_ctx)</label>
      <input type="number" name="num_ctx" value="8192" min="1024" max="65536" step="256" />
    </div>

    <div class="row">
      <label>Limit objectives</label>
      <input type="number" name="limit" value="0" min="0" max="9999" />
    </div>

    <button class="btn" type="submit">Generate</button>
  </form>

  {% if job %}
    <div class="card">
      <h2>Job: {{ job_id }}</h2>
      <p><b>Status:</b> {{ job.status }}</p>
      <p><b>Requested:</b> {{ job.requested }} | <b>Completed:</b> {{ job.completed }} | <b>Failures:</b> {{ job.failures|length }}</p>
      <div class="links">
        {% if job.status == 'done' %}
          <a href="/download/master/{{ job_id }}">Download master .txt</a>
          <a href="/download/zip/{{ job_id }}">Download per-objective ZIP</a>
        {% endif %}
      </div>

      {% if job.failures|length > 0 %}
        <h3>Failures</h3>
        <pre>{{ job.failures | join('\\n') }}</pre>
      {% endif %}
    </div>
  {% endif %}
</div>
</body>
</html>
"""

@app.get("/")
def home():
    return render_template_string(PAGE, job=None, job_id=None)

@app.get("/job/<job_id>")
def job_view(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        abort(404)
    return render_template_string(PAGE, job=job, job_id=job_id)

@app.post("/generate")
def generate():
    up = request.files.get("input_file")
    if not up or up.filename == "":
        log_progress("ERROR: No file uploaded")
        return "No file uploaded", 400

    log_progress("=" * 60)
    log_progress("NEW GENERATION REQUEST RECEIVED")
    log_progress("=" * 60)

    # Read file content
    content = up.read().decode("utf-8", errors="replace")
    log_progress(f"File uploaded: {up.filename} ({len(content)} bytes)")

    ollama_url = request.form.get("ollama_url", "http://localhost:11434").strip()
    worker_model = request.form.get("worker_model", "qwen:32b").strip()
    manager_model = request.form.get("manager_model", "qwen:32b").strip()
    max_rounds = int(request.form.get("max_rounds", "6"))
    num_ctx = int(request.form.get("num_ctx", "8192"))
    limit = int(request.form.get("limit", "0"))

    log_progress(f"Configuration:")
    log_progress(f"  Ollama URL: {ollama_url}")
    log_progress(f"  Worker model: {worker_model}")
    log_progress(f"  Manager model: {manager_model}")
    log_progress(f"  Max rounds: {max_rounds}")
    log_progress(f"  Context size: {num_ctx}")
    log_progress(f"  Limit: {limit if limit > 0 else 'None'}")

    job_id = str(uuid.uuid4())[:8]
    log_progress(f"Job ID: {job_id}")
    
    JOBS[job_id] = {
        "status": "running",
        "requested": 0,
        "completed": 0,
        "failures": [],
        "master_text": "",
        "zip_bytes": b"",
        "meta": {
            "ollama_url": ollama_url,
            "worker_model": worker_model,
            "manager_model": manager_model,
            "max_rounds": max_rounds,
            "num_ctx": num_ctx,
            "limit": limit,
        }
    }

    # Run synchronously (local app). If you want async later, we can add Celery/RQ.
    try:
        log_progress("Parsing objectives from file...")
        roots = parse_objectives(content)
        objectives = collect_learning_objectives_with_paths(roots)
        if limit and limit > 0:
            objectives = objectives[:limit]
            log_progress(f"Limited to first {limit} objectives")

        total_objectives = len(objectives)
        JOBS[job_id]["requested"] = total_objectives
        log_progress(f"Found {total_objectives} learning objectives to process")
        log_progress("-" * 60)

        generated_by_id: Dict[str, Dict[str, Any]] = {}
        failures: List[str] = []

        # Prepare zip in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # index content built as we go
            index_lines = []
            index_lines.append("IKIZAMINI PER-OBJECTIVE INDEX (Ollama UI)")
            index_lines.append(f"Ollama URL: {ollama_url}")
            index_lines.append(f"Worker model: {worker_model}")
            index_lines.append(f"Manager model: {manager_model}")
            index_lines.append(f"Generated at (unix): {int(time.time())}")
            index_lines.append("")

            for i, obj in enumerate(objectives, start=1):
                obj_id = obj["objective_id"]
                obj_text = obj["objective_text"]
                path = obj["path"]
                
                log_progress(f"[{i}/{total_objectives}] Processing: {obj_id} - {obj_text[:60]}...")
                start_time = time.time()
                
                try:
                    ik = generate_for_objective_strict(
                        base_url=ollama_url,
                        worker_model=worker_model,
                        manager_model=manager_model,
                        objective_id=obj_id,
                        objective_text=obj_text,
                        max_rounds=max_rounds,
                        num_ctx=num_ctx,
                    )
                    validate_ikizamini(ik)
                    generated_by_id[obj_id] = ik
                    elapsed = time.time() - start_time
                    log_progress(f"  ✓ Success ({elapsed:.1f}s)")
                    
                    JOBS[job_id]["completed"] = len(generated_by_id)

                    fname = unique_objective_filename(obj_id, obj_text)
                    txt = objective_file_text(path, obj_id, obj_text, ik, None)
                    zf.writestr(fname, txt)
                    index_lines.append(f"- {fname}")

                except Exception as e:
                    elapsed = time.time() - start_time
                    msg = f"{obj_id}: {e}"
                    log_progress(f"  ✗ FAILED ({elapsed:.1f}s): {msg}")
                    failures.append(msg)
                    JOBS[job_id]["failures"] = failures

                    fname = unique_objective_filename(obj_id, obj_text)
                    txt = objective_file_text(path, obj_id, obj_text, None, str(e))
                    zf.writestr(fname, txt)
                    index_lines.append(f"- {fname}  (FAILED)")

            # write index
            zf.writestr("index.txt", "\n".join(index_lines).rstrip() + "\n")

        zip_bytes = zip_buffer.getvalue()

        meta = {
            "ollama_url": ollama_url,
            "worker_model": worker_model,
            "manager_model": manager_model,
            "generated_at_unix": int(time.time()),
            "requested": len(objectives),
            "completed": len(generated_by_id),
        }
        
        log_progress("-" * 60)
        log_progress("Generating master file...")
        master_text = write_master_from_tree(roots, generated_by_id, meta)

        JOBS[job_id]["master_text"] = master_text
        JOBS[job_id]["zip_bytes"] = zip_bytes
        JOBS[job_id]["failures"] = failures
        JOBS[job_id]["completed"] = len(generated_by_id)
        JOBS[job_id]["status"] = "done"

        log_progress("=" * 60)
        log_progress(f"JOB COMPLETED: {job_id}")
        log_progress(f"  Total: {total_objectives} | Success: {len(generated_by_id)} | Failed: {len(failures)}")
        log_progress("=" * 60)

    except Exception as e:
        log_progress(f"ERROR: {str(e)}")
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["failures"] = [str(e)]

    return redirect(url_for("job_view", job_id=job_id))

@app.get("/download/master/<job_id>")
def download_master(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        abort(404)
    data = job["master_text"].encode("utf-8")
    return send_file(
        io.BytesIO(data),
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"ikizamini_master_{job_id}.txt",
    )

@app.get("/download/zip/<job_id>")
def download_zip(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        abort(404)
    return send_file(
        io.BytesIO(job["zip_bytes"]),
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"ikizamini_outputs_{job_id}.zip",
    )

if __name__ == "__main__":
    # Flask dev server
    # For Runpod: use 0.0.0.0 and port 8000 (or from RUNPOD_PORT env var)
    # For local: use 127.0.0.1 and port 5000
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("RUNPOD_PORT", os.environ.get("FLASK_PORT", "8000")))
    app.run(host=host, port=port, debug=False)
