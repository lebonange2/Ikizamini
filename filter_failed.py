#!/usr/bin/env python3
"""
Filter lines containing "[failed]" from a text file and write them to another file.

Usage:
  python3 filter_failed.py "Finished/ikizamini_outputs_fba892ef/outputs/index.txt" failed.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def filter_failed_lines(input_path: Path, output_path: Path, needle: str = "[failed]") -> int:
    """
    Returns the number of lines written to output_path.
    """
    written = 0

    # Ensure parent directory exists (if output includes a directory)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", errors="replace") as fin, \
         output_path.open("w", encoding="utf-8", newline="") as fout:
        for line in fin:
            if needle in line:
                fout.write(line)
                written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract lines containing "[failed]" into a new file.')
    parser.add_argument("input_file", type=Path, help="Path to the input text file.")
    parser.add_argument("output_file", type=Path, help="Path to write the filtered lines.")
    args = parser.parse_args()

    if not args.input_file.exists():
        raise SystemExit(f"Input file not found: {args.input_file}")

    count = filter_failed_lines(args.input_file, args.output_file)
    print(f"Wrote {count} failed line(s) to: {args.output_file}")


if __name__ == "__main__":
    main()
