IKIZAMINI
=========

IKIZAMINI generates exam-style question sets per learning objective from an outline file.

This repo includes:
- `ikizamini_local.py`: Flask UI that talks to **Ollama**
- `ikizamini_app.py`: CLI version (OpenAI Responses API client)
- `setup.sh`: Runpod-friendly setup (installs deps, installs Ollama, pulls Qwen model, starts UI)
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
- `http://127.0.0.1:8000` (local)
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