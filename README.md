IKIZAMINI
=========

IKIZAMINI generates exam-style question sets per learning objective from an outline file.

This repo includes:
- `ikizamini_local.py`: Flask UI that talks to **Ollama**
- `ikizamini_app.py`: CLI version (OpenAI Responses API client)
- `setup.sh`: Runpod-friendly setup (installs deps, installs Ollama, pulls Qwen model, starts UI)
- `ikizamini_local_parallel.py`: Flask UI (same UI as `ikizamini_local.py`) but jobs process objectives concurrently
- `ikizamini_local_parallel_cli.py`: parallel/concurrent CLI runner for Ollama (saves to `Parallely_Processed/`)
- `remove_json_files.py`: cleanup utility to remove `*.json` files from the output folder

## Quick start (Runpod / cloud)

Run:

```bash
./setup.sh
```

What it does:
- Installs `zstd` (system package)
- Creates/uses `venv/` and installs Python deps (`flask`, `requests`, `jsonschema`, `openai`)
- Installs **Ollama**
- Pulls a Qwen model (tries `qwen:32b` first; falls back to other variants)
- Starts the UI on `0.0.0.0:8000` (or `RUNPOD_PORT` if set)

## Running the UI (Ollama)

If you prefer to run manually:

```bash
source venv/bin/activate
ollama serve &
python3 ikizamini_local.py
```

Then open:
- `http://127.0.0.1:8000` (local; may auto-pick `8001` if `8000` is busy)
- On Runpod, use the public URL mapped to port 8000 (or `RUNPOD_PORT`)

### Notes
- The UI prints detailed progress logs to the terminal for each objective.
- The Generate button shows a loading state immediately so you know it was clicked.
- The app is strict about schema output and includes sanitization to handle common model key-typos (unicode confusables, punctuation, etc.).

## Running the CLI (OpenAI)

```bash
python3 ikizamini_app.py \
  --input Uru.txt \
  --output ikizamini.txt \
  --output-dir "output/MATHEMATICS/1.1 Algebra and Trigonometry"
```

## Running the parallel UI (Ollama) — `ikizamini_local_parallel.py`

This starts the same UI as `ikizamini_local.py`, but each job processes objectives concurrently.

```bash
source venv/bin/activate
export IKIZAMINI_PARALLEL_WORKERS=4
python3 ikizamini_local_parallel.py
```

## Running the parallel CLI (Ollama) — `ikizamini_local_parallel_cli.py`

This runner processes learning objectives concurrently (single-process threads) while Ollama does the heavy work.

### Smoke-check (fast validation)

Checks that Ollama is reachable and the model can respond with JSON (no `--input` needed):

```bash
source venv/bin/activate
python3 ikizamini_local_parallel_cli.py --smoke-check --ollama-url http://localhost:11434 --worker-model gemma2:2b
```

### Generate outputs

```bash
source venv/bin/activate
python3 ikizamini_local_parallel_cli.py \
  --input Uru.txt \
  --ollama-url http://localhost:11434 \
  --worker-model gemma3:latest \
  --manager-model gemma3:latest \
  --workers 4 \
  --output-dir Parallely_Processed
```

Useful flags:
- `--limit 10`: only process the first 10 learning objectives (good for testing)
- `--timeout-s 600`: increase Ollama request timeout for slower models/CPU runs
- `--max-rounds 6`: repair/review rounds per objective
- `--num-ctx 8192`: context size

Outputs:
- Per objective: `Parallely_Processed/<objective>_<hash>.json` and `.txt`
- Index: `Parallely_Processed/index.txt`

## Removing JSON files from the output directory

To remove `*.json` files under:
`output/MATHEMATICS/1.1 Algebra and Trigonometry/`

### Dry run (recommended first)

```bash
python3 remove_json_files.py
```

### Actually delete

```bash
python3 remove_json_files.py --yes
```

### Custom directory

```bash
python3 remove_json_files.py --dir "output/MATHEMATICS/1.1 Algebra and Trigonometry" --yes
```