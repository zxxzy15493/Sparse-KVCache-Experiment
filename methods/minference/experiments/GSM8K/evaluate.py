#!/usr/bin/env python3
"""Extract GSM8K answers from `pred` and evaluate accuracy.

Many prediction jsonl files in this repo include a full model output in `pred`,
but may leave `match_result` empty/null. This script:

  1) reads a GSM8K jsonl file line-by-line
  2) extracts the predicted final answer from the `pred` field
  3) writes it into `match_result`
  4) computes score = correct / total and prints (score, correct, total)

Comparison uses numeric-aware normalization (supports ints/floats/fractions,
commas, and common formatting like \\boxed{...}).

By default it updates the input file in-place and creates a `.bak` backup.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

BOXED_RE = re.compile(r"\\boxed\s*\{|\bboxed\s*\{", re.IGNORECASE)

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_FRACTION_RE = re.compile(r"(-?\d+)\s*/\s*(\d+)")
_GOLD_HASH_RE = re.compile(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def has_boxed(text: str) -> bool:
    return bool(BOXED_RE.search(text or ""))


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
    """Parse numeric-like strings (int/float/fraction) into Decimal."""

    v = _strip_and_collapse_spaces(value)
    if not v:
        return None

    m = _FRACTION_RE.fullmatch(v)
    if m:
        try:
            frac = Fraction(int(m.group(1)), int(m.group(2)))
        except Exception:
            return None
        return Decimal(frac.numerator) / Decimal(frac.denominator)

    v = v.replace(",", "")
    try:
        return Decimal(v)
    except InvalidOperation:
        return None


def normalize_answer(value: Any) -> str:
    """Normalize an answer-like value into a comparable string.

    Extracts the last number-like token if present; otherwise collapses spaces.
    """

    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        value = str(value)
    s = value.strip()
    if not s:
        return ""

    # If the whole string is a fraction, keep it.
    if _FRACTION_RE.fullmatch(_strip_and_collapse_spaces(s)):
        return _strip_and_collapse_spaces(s)

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
        return abs(pred_num - gold_num) <= Decimal("1e-9")

    return _strip_and_collapse_spaces(pred_norm).lower() == _strip_and_collapse_spaces(gold_norm).lower()


def extract_gold_from_answer(answer_text: Any) -> str:
    """Extract gold answer from GSM8K `answer` field.

    GSM8K uses a canonical line `#### <number>`.
    """

    if answer_text is None:
        return ""
    if not isinstance(answer_text, str):
        answer_text = str(answer_text)

    matches = _GOLD_HASH_RE.findall(answer_text)
    if matches:
        return matches[-1].replace(",", "")
    return normalize_answer(answer_text)


def _extract_balanced_braces(text: str, open_brace_index: int) -> Optional[tuple[str, int]]:
    """Return (content, end_index) for a {...} group starting at '{'.

    `end_index` is the index of the matching '}' (inclusive).
    """

    if open_brace_index < 0 or open_brace_index >= len(text) or text[open_brace_index] != "{":
        return None
    depth = 0
    i = open_brace_index
    start = open_brace_index + 1
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i], i
        i += 1
    return None


def extract_predicted_answer_from_pred(pred_text: Any) -> str:
    """Extract a predicted final answer string from the full `pred` text."""

    if pred_text is None:
        return ""
    if not isinstance(pred_text, str):
        pred_text = str(pred_text)
    s = pred_text.strip()
    if not s:
        return ""

    # Remove time-like tokens (e.g., 1:00, 12:34) to avoid picking trailing "00".
    s_no_times = re.sub(r"\b\d{1,2}:\d{2}\b", " ", s)

    # 1) <answer>...</answer>
    m = _ANSWER_TAG_RE.search(s)
    if m:
        return normalize_answer(m.group(1))

    # 2) LaTeX \boxed{...}
    last_boxed_content: Optional[str] = None
    idx = 0
    while True:
        j = s.find("\\boxed", idx)
        if j == -1:
            break
        k = s.find("{", j)
        if k == -1:
            idx = j + 6
            continue
        extracted = _extract_balanced_braces(s, k)
        if extracted is None:
            idx = k + 1
            continue
        content, end_idx = extracted
        last_boxed_content = content
        idx = end_idx + 1
    if last_boxed_content is not None:
        cleaned = last_boxed_content
        cleaned = cleaned.replace("\\$", "$")
        cleaned = cleaned.replace("$", "")
        cleaned = re.sub(r"\\!|\\,|\\;|\\:|\\\s", "", cleaned)
        return normalize_answer(cleaned)

    # 3) GSM8K-style #### line inside prediction
    matches = _GOLD_HASH_RE.findall(s)
    if matches:
        return matches[-1].replace(",", "")

    # 4) Marker-based extraction
    marker_re = re.compile(
        r"(?:final\s+answer|answer)\s*[:：]", re.IGNORECASE
    )
    last_marker = None
    for m2 in marker_re.finditer(s_no_times):
        last_marker = m2
    if last_marker is not None:
        tail = s_no_times[last_marker.end():]
        m_frac = _FRACTION_RE.search(tail)
        if m_frac:
            return f"{m_frac.group(1)}/{m_frac.group(2)}"
        nums = _NUMBER_RE.findall(tail)
        if nums:
            return nums[0].replace(",", "")

    # 5) Fallback: last number-like token in the whole output
    return normalize_answer(s_no_times)


@dataclass
class EvalResult:
    total: int
    correct: int

    @property
    def score(self) -> float:
        return 0.0 if self.total == 0 else self.correct / self.total

    @property
    def wrong(self) -> int:
        return self.total - self.correct


def evaluate_jsonl(
    input_path: Path,
    *,
    output_path: Path,
    error_path: Path,
    error_and_no_boxed_path: Path,
    force: bool,
    write_correct_field: bool,
) -> EvalResult:
    total = 0
    correct = 0

    with (
        input_path.open("r", encoding="utf-8") as f_in,
        output_path.open("w", encoding="utf-8") as f_out,
        error_path.open("w", encoding="utf-8") as f_err,
        error_and_no_boxed_path.open("w", encoding="utf-8") as f_err_and_no_boxed,
    ):
        for line_no, line in enumerate(f_in, start=1):
            obj = _safe_json_loads(line, path=input_path, line_no=line_no)
            if obj is None:
                continue

            pred_extracted = extract_predicted_answer_from_pred(obj.get("pred"))

            existing = obj.get("match_result")
            if force or existing is None or (isinstance(existing, str) and not existing.strip()):
                obj["match_result"] = pred_extracted

            # Determine gold answer
            gold = obj.get("final_answer")
            if gold is None or (isinstance(gold, str) and not gold.strip()):
                gold = extract_gold_from_answer(obj.get("answer"))
                if gold:
                    obj["final_answer"] = gold
            gold_norm = normalize_answer(gold)

            pred_norm = normalize_answer(obj.get("match_result"))
            ok = is_correct(pred_norm, gold_norm)

            total += 1
            if ok:
                correct += 1
            if write_correct_field:
                obj["correct"] = ok
            if not ok:
                json.dump(obj, f_err, ensure_ascii=False)
                f_err.write("\n")
                if not has_boxed(obj.get("pred", "")):
                    json.dump(obj, f_err_and_no_boxed, ensure_ascii=False)
                    f_err_and_no_boxed.write("\n")

            json.dump(obj, f_out, ensure_ascii=False)
            f_out.write("\n")

    return EvalResult(total=total, correct=correct)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract GSM8K final answers from pred and compute score.")
    p.add_argument("--input", required=True, help="Path to gsm8k.jsonl")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write error.jsonl and result.jsonl into (default: input file directory).",
    )
    p.add_argument("--output", default=None, help="Output jsonl path. If omitted, updates input in-place.")
    p.add_argument("--force", action="store_true", help="Overwrite existing match_result even if non-empty.")
    p.add_argument(
        "--write-correct-field",
        action="store_true",
        help="Also write per-line boolean field `correct` into output jsonl.",
    )
    return p.parse_args(argv)


def write_result_json(
    result_path: Path, *, input_path: Path, output_path: Optional[Path], result: EvalResult
) -> None:
    payload = {
        "metric": "accuracy",
        "score": result.score,
        "correct": result.correct,
        "wrong": result.wrong,
        "total": result.total,
        "input": str(input_path),
        "output": str(output_path) if output_path is not None else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
        f.write("\n")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input does not exist: {input_path}")

    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    error_path = out_dir / "error.jsonl"
    error_and_no_boxed_path = out_dir / "error_and_no_boxed.jsonl"
    result_path = out_dir / "result.json"

    if args.output:
        output_path = Path(args.output)
        result = evaluate_jsonl(
            input_path,
            output_path=output_path,
            error_path=error_path,
            error_and_no_boxed_path=error_and_no_boxed_path,
            force=args.force,
            write_correct_field=args.write_correct_field,
        )
        write_result_json(result_path, input_path=input_path, output_path=output_path, result=result)
        print(
            f"score={result.score:.6f}  correct={result.correct}/{result.total}  "
            f"(wrong={result.wrong})  -> {output_path}"
        )
        print(f"wrote: {error_path}")
        print(f"wrote: {error_and_no_boxed_path}")
        print(f"wrote: {result_path}")
        return 0

    # In-place update with backup
    backup_path = input_path.with_suffix(input_path.suffix + ".bak")
    tmp_path = input_path.with_suffix(input_path.suffix + ".tmp")
    shutil.copy2(input_path, backup_path)
    result = evaluate_jsonl(
        input_path,
        output_path=tmp_path,
        error_path=error_path,
        error_and_no_boxed_path=error_and_no_boxed_path,
        force=args.force,
        write_correct_field=args.write_correct_field,
    )
    tmp_path.replace(input_path)
    write_result_json(result_path, input_path=input_path, output_path=input_path, result=result)

    print(f"score={result.score:.6f}  correct={result.correct}/{result.total}  (backup: {backup_path})")
    print(f"wrote: {error_path}")
    print(f"wrote: {error_and_no_boxed_path}")
    print(f"wrote: {result_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as e:
        print(f"[evaluate.py] ERROR: {e}", file=sys.stderr)
        raise
