#!/usr/bin/env python3
"""
remove_json_files.py

Removes (recursively) all .json files from:
  output/MATHEMATICS/1.1 Algebra and Trigonometry

Safety:
- Default is DRY RUN (prints what would be deleted)
- Use --yes to actually delete

Examples:
  python3 remove_json_files.py
  python3 remove_json_files.py --yes
  python3 remove_json_files.py --dir "output/MATHEMATICS/1.5 Calculus (Single Variable)" --yes
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_DIR = Path("output") / "MATHEMATICS" / "1.1 Algebra and Trigonometry"


def iter_json_files(root: Path) -> list[Path]:
    # Case-insensitive match: *.json, *.JSON, etc.
    files: list[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".json":
            files.append(p)
    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser(description="Remove .json files from an output folder (recursively).")
    ap.add_argument(
        "--dir",
        default=str(DEFAULT_DIR),
        help=f"Target directory (default: {DEFAULT_DIR})",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files (without this flag, runs in dry-run mode).",
    )
    args = ap.parse_args()

    root = Path(args.dir)
    json_files = iter_json_files(root)

    if not root.exists():
        print(f"[ERROR] Directory does not exist: {root}")
        return 2

    if not json_files:
        print(f"[OK] No .json files found under: {root}")
        return 0

    mode = "DELETE" if args.yes else "DRY RUN"
    print(f"[{mode}] Found {len(json_files)} .json files under: {root}")
    for p in json_files:
        print(f" - {p}")

    if not args.yes:
        print("\nNothing deleted (dry-run). Re-run with --yes to delete.")
        return 0

    deleted = 0
    failed = 0
    for p in json_files:
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] Failed to delete {p}: {e}")

    print(f"\n[RESULT] Deleted: {deleted} | Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

