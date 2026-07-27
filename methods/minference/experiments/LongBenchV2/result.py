"""Evaluate LongBenchV2 prediction results.

Computes accuracy broken down by difficulty (easy/hard) and length (short/medium/long),
and writes a formatted table to result.txt.

Usage:
    python result.py [results_dir] [output_file]

    results_dir: path to directory containing prediction JSONL files
    output_file: path to save results table (default: result.txt)
"""

import os
import sys
import json
from collections import defaultdict

from utils import load_data


def evaluate_file(filepath):
    """Evaluate a single prediction JSONL file.

    Returns a dict with keys: overall, easy, hard, short, medium, long.
    Each value is (correct, total).
    """
    counts = {
        "overall": [0, 0],
        "easy": [0, 0],
        "hard": [0, 0],
        "short": [0, 0],
        "medium": [0, 0],
        "long": [0, 0],
    }

    for sample in load_data(filepath):
        judge = sample.get("judge", False)
        difficulty = sample.get("difficulty", "")
        length = sample.get("length", "")

        counts["overall"][1] += 1
        if judge:
            counts["overall"][0] += 1

        if difficulty in ("easy", "hard"):
            counts[difficulty][1] += 1
            if judge:
                counts[difficulty][0] += 1

        if length in ("short", "medium", "long"):
            counts[length][1] += 1
            if judge:
                counts[length][0] += 1

    return counts


def format_pct(correct, total):
    if total == 0:
        return "N/A"
    return f"{correct / total * 100:.1f}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python result.py <results_dir> [output_file]")
        print("Example: python result.py results/Qwen2.5-7B-Instruct-1M/")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "result.txt"

    if not os.path.isdir(results_dir):
        raise NotADirectoryError(f"Not a directory: {results_dir}")

    # Collect all JSONL files
    jsonl_files = sorted([
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".jsonl")
    ])

    if not jsonl_files:
        print(f"No JSONL files found in {results_dir}")
        sys.exit(1)

    lines = []
    header = "Model\tOverall\tEasy\tHard\tShort\tMedium\tLong"
    lines.append(header)

    for filepath in jsonl_files:
        name = os.path.splitext(os.path.basename(filepath))[0]
        counts = evaluate_file(filepath)
        row = "\t".join([
            name,
            format_pct(*counts["overall"]),
            format_pct(*counts["easy"]),
            format_pct(*counts["hard"]),
            format_pct(*counts["short"]),
            format_pct(*counts["medium"]),
            format_pct(*counts["long"]),
        ])
        lines.append(row)

    result_text = "\n".join(lines) + "\n"
    with open(output_file, "w") as f:
        f.write(result_text)

    print(result_text)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
