#!/usr/bin/env python3
"""Evaluate GSM8K prediction jsonl(s) and report accuracy.

Input jsonl line format (at minimum):
  {"match_result": "model predicted result", "final_answer": "gold answer", ...}

This script computes score (= accuracy) by comparing normalized predicted answer
against normalized gold answer.

Outputs:
  {output_dir}/result.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Optional


_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_FRACTION_RE = re.compile(r"(-?\d+)\s*/\s*(\d+)")


def _safe_json_loads(line: str, *, path: Path, line_no: int) -> Optional[dict[str, Any]]:
	line = line.strip()
	if not line:
		return None
	try:
		obj = json.loads(line)
	except json.JSONDecodeError as e:
		raise ValueError(f"Invalid JSON in {path} at line {line_no}: {e}") from e
	if not isinstance(obj, dict):
		raise ValueError(f"JSON line is not an object in {path} at line {line_no}")
	return obj


def _strip_and_collapse_spaces(s: str) -> str:
	return " ".join(s.strip().split())


def _to_decimal_like(value: str) -> Optional[Decimal]:
	"""Parse numeric-like strings (int/float/fraction) into Decimal.

	Returns None if parsing is not possible.
	"""

	v = _strip_and_collapse_spaces(value)
	if not v:
		return None

	# Fraction like 3/4
	m = _FRACTION_RE.fullmatch(v)
	if m:
		try:
			frac = Fraction(int(m.group(1)), int(m.group(2)))
		except Exception:
			return None
		return Decimal(frac.numerator) / Decimal(frac.denominator)

	# Remove thousands separators
	v = v.replace(",", "")
	try:
		return Decimal(v)
	except InvalidOperation:
		return None


def normalize_prediction(match_result: Any) -> str:
	"""Normalize model output field `match_result` into a comparable answer string."""

	if match_result is None:
		return ""
	if isinstance(match_result, (int, float)):
		return str(match_result)
	if not isinstance(match_result, str):
		match_result = str(match_result)

	s = match_result.strip()
	if not s:
		return ""

	# Prefer a standalone fraction if the whole string is a fraction
	if _FRACTION_RE.fullmatch(_strip_and_collapse_spaces(s)):
		return _strip_and_collapse_spaces(s)

	# Extract last number-like token from the string (GSM8K answers are typically at the end).
	nums = _NUMBER_RE.findall(s)
	if not nums:
		# Fallback: collapse spaces and strip common wrappers
		return _strip_and_collapse_spaces(s)

	return nums[-1].replace(",", "")


def normalize_gold(final_answer: Any) -> str:
	if final_answer is None:
		return ""
	if isinstance(final_answer, (int, float)):
		return str(final_answer)
	if not isinstance(final_answer, str):
		final_answer = str(final_answer)
	s = final_answer.strip()

	# Some pipelines may include the '#### ' prefix; strip it if present.
	if s.startswith("####"):
		s = s.lstrip("#").strip()

	# If gold is not purely a number, still try to extract a number.
	nums = _NUMBER_RE.findall(s)
	if nums:
		return nums[-1].replace(",", "")

	return _strip_and_collapse_spaces(s)


def is_correct(pred_norm: str, gold_norm: str) -> bool:
	if not pred_norm or not gold_norm:
		return False

	pred_num = _to_decimal_like(pred_norm)
	gold_num = _to_decimal_like(gold_norm)
	if pred_num is not None and gold_num is not None:
		# Exact compare is usually OK for integers; still use a tiny tolerance.
		return abs(pred_num - gold_num) <= Decimal("1e-9")

	return _strip_and_collapse_spaces(pred_norm).lower() == _strip_and_collapse_spaces(gold_norm).lower()


@dataclass
class FileResult:
	file: str
	total: int
	correct: int

	@property
	def score(self) -> float:
		return 0.0 if self.total == 0 else self.correct / self.total


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
	with path.open("r", encoding="utf-8") as f:
		for i, line in enumerate(f, start=1):
			obj = _safe_json_loads(line, path=path, line_no=i)
			if obj is not None:
				yield obj


def evaluate_file(path: Path) -> FileResult:
	total = 0
	correct = 0

	for obj in iter_jsonl(path):
		total += 1
		pred_norm = normalize_prediction(obj.get("match_result"))
		gold_norm = normalize_gold(obj.get("final_answer"))
		if is_correct(pred_norm, gold_norm):
			correct += 1

	return FileResult(file=str(path), total=total, correct=correct)


def find_jsonl_files(input_dir: Path) -> list[Path]:
	if not input_dir.exists():
		raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
	if not input_dir.is_dir():
		raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

	# Prefer direct children; if none found, fall back to recursive search.
	direct = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix == ".jsonl"])
	if direct:
		return direct

	recursive = sorted([p for p in input_dir.rglob("*.jsonl") if p.is_file()])
	return recursive


def parse_args(argv: list[str]) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate GSM8K jsonl predictions and compute accuracy.")

	# Support both optional flags and positional args.
	parser.add_argument("input_dir", nargs="?", help="Directory that contains prediction jsonl file(s).")
	parser.add_argument("output_dir", nargs="?", help="Directory to write result.json into.")
	parser.add_argument("--input_dir", dest="input_dir_flag", help="Same as positional input_dir.")
	parser.add_argument("--output_dir", dest="output_dir_flag", help="Same as positional output_dir.")

	ns = parser.parse_args(argv)
	input_dir = ns.input_dir_flag or ns.input_dir
	output_dir = ns.output_dir_flag or ns.output_dir
	if not input_dir or not output_dir:
		parser.error("Requires input_dir and output_dir (positional or via --input_dir/--output_dir)")
	ns.input_dir = input_dir
	ns.output_dir = output_dir
	return ns


def main(argv: list[str]) -> int:
	args = parse_args(argv)
	input_dir = Path(args.input_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	jsonl_files = find_jsonl_files(input_dir)
	if not jsonl_files:
		raise FileNotFoundError(f"No .jsonl files found under input_dir: {input_dir}")

	file_results: list[FileResult] = [evaluate_file(p) for p in jsonl_files]
	total = sum(fr.total for fr in file_results)
	correct = sum(fr.correct for fr in file_results)
	score = 0.0 if total == 0 else correct / total

	result = {
		"metric": "accuracy",
		"score": score,
		"correct": correct,
		"total": total,
		"files": [
			{"file": fr.file, "score": fr.score, "correct": fr.correct, "total": fr.total}
			for fr in file_results
		],
		"input_dir": os.fspath(input_dir),
		"generated_at": datetime.now(timezone.utc).isoformat(),
	}

	out_path = output_dir / "result.json"
	with out_path.open("w", encoding="utf-8") as f:
		json.dump(result, f, ensure_ascii=False, indent=2)
		f.write("\n")

	print(f"score={score:.6f}  correct={correct}/{total}  -> {out_path}")
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main(sys.argv[1:]))
	except Exception as e:
		print(f"[evaluate.py] ERROR: {e}", file=sys.stderr)
		raise

