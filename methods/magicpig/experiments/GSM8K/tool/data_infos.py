
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BOXED_RE = re.compile(r"\\boxed\s*\{|\bboxed\s*\{", re.IGNORECASE)


@dataclass(frozen=True)
class BinSpec:
    name: str
    left_inclusive: int
    right_exclusive: Optional[int]  # None means +inf

    def contains(self, value: int) -> bool:
        if value < self.left_inclusive:
            return False
        if self.right_exclusive is None:
            return True
        return value < self.right_exclusive


BIN_SPECS: List[BinSpec] = [
    BinSpec("[0,1000)", 0, 1000),
    BinSpec("[1000,1001)", 1000, 1001),
    BinSpec("[1001,3000)", 1001, 3000),
    BinSpec("[3000,6000)", 3000, 6000),
    BinSpec("[6000,9000)", 6000, 9000),
    BinSpec("[9000,20000)", 9000, 20000),
    BinSpec("20000+", 20000, None),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def has_boxed(text: str) -> bool:
    return bool(BOXED_RE.search(text or ""))


def _iter_jsonl(path: Path) -> Iterable[Tuple[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            yield i, line.rstrip("\n")


def _resolve_data_file(data_dir: Path, task: str) -> Path:
    """Resolve the input JSONL file.

    Per requirement: input file is `{data_dir}/{task}.jsonl`.
    - If `task` already ends with `.jsonl`, it will not be duplicated.
    """
    filename = task if task.endswith(".jsonl") else f"{task}.jsonl"
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def _token_length(tokenizer, text: str) -> int:
    if text is None:
        return 0
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        return 0
    # input_ids can be list[int] or list[list[int]]
    if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _bin_name(length: int) -> str:
    for spec in BIN_SPECS:
        if spec.contains(length):
            return spec.name
    # Should be unreachable.
    return BIN_SPECS[-1].name


def _bin_index(length: int) -> int:
    for i, spec in enumerate(BIN_SPECS):
        if spec.contains(length):
            return i
    return len(BIN_SPECS) - 1


def analyze_file(eval_path: Path, *, tokenizer_name_or_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Missing dependency 'transformers'. Please run under your 'kvcache' conda env, "
            "or install transformers into it (e.g., `pip install transformers`)."
        ) from e

    # Prefer local cache to avoid slow/unstable network; fall back to online fetch if needed.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

    # Avoid warnings when we only need lengths (no model forward pass).
    try:
        if getattr(tokenizer, "model_max_length", None) and tokenizer.model_max_length < 10**9:
            tokenizer.model_max_length = 10**9
    except Exception:  # noqa: BLE001
        pass

    bin_counts: Dict[str, int] = {b.name: 0 for b in BIN_SPECS}
    total_json = 0
    total_with_pred = 0
    total_without_boxed = 0
    invalid_json_lines = 0
    missing_pred_field = 0

    total_token_len_sum = 0
    total_token_len_count = 0

    bin_sums = [0] * len(BIN_SPECS)
    bin_for_data_index: Dict[str, List] = {b.name: [] for b in BIN_SPECS}
    bin_counts_len = [0] * len(BIN_SPECS)

    no_boxed_rows: List[Dict[str, Any]] = []

    for line_no, raw in _iter_jsonl(eval_path):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            invalid_json_lines += 1
            no_boxed_rows.append({"__raw_line__": raw, "__line__": line_no, "__error__": str(e)})
            continue

        total_json += 1

        pred = row.get("pred")
        if isinstance(pred, str):
            total_with_pred += 1
            length = _token_length(tokenizer, pred)
            bin_counts[_bin_name(length)] += 1
            
            bin_for_data_index[_bin_name(length)].append(row['index'])

            total_token_len_sum += length
            total_token_len_count += 1

            bidx = _bin_index(length)
            bin_sums[bidx] += length
            bin_counts_len[bidx] += 1
        else:
            missing_pred_field += 1

        pred_text = pred if isinstance(pred, str) else ""
        if not has_boxed(pred_text):
            total_without_boxed += 1
            no_boxed_rows.append(row)

    summary: Dict[str, Any] = {
        "created_at": _utc_now_iso(),
        "data_file": str(eval_path),
        "tokenizer": tokenizer_name_or_path,
        "bin_counts": bin_counts,
        "bin_for_data_index": bin_for_data_index,
        "total_json": total_json,
        "total_with_pred": total_with_pred,
        "missing_pred_field": missing_pred_field,
        "total_without_boxed": total_without_boxed,
        "invalid_json_lines": invalid_json_lines,
        "avg_token_length": (total_token_len_sum / total_token_len_count) if total_token_len_count else None,
        "avg_token_len_pre_one": (bin_sums[0] / bin_counts_len[0]) if bin_counts_len[0] else None,
        "avg_token_len_pre_two": (
            (sum(bin_sums[:2]) / sum(bin_counts_len[:2])) if sum(bin_counts_len[:2]) else None
        ),
        "avg_token_len_pre_three": (
            (sum(bin_sums[:3]) / sum(bin_counts_len[:3])) if sum(bin_counts_len[:3]) else None
        ),
        "avg_token_len_pre_four": (
            (sum(bin_sums[:4]) / sum(bin_counts_len[:4])) if sum(bin_counts_len[:4]) else None
        ),
        "avg_token_len_post_one": (
            (bin_sums[-1] / bin_counts_len[-1]) if bin_counts_len[-1] else None
        ),
        "avg_token_len_post_two": (
            (sum(bin_sums[-2:]) / sum(bin_counts_len[-2:])) if sum(bin_counts_len[-2:]) else None
        ),
        "avg_token_len_by_bin": {
            BIN_SPECS[i].name: ((bin_sums[i] / bin_counts_len[i]) if bin_counts_len[i] else None)
            for i in range(len(BIN_SPECS))
        },
    }
    return summary, no_boxed_rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect token-length bins and boxed{} coverage for a JSONL file.")
    parser.add_argument("--data-dir", required=True, help="Data directory")
    parser.add_argument("--model", required=True, help="Model name/path for AutoTokenizer.from_pretrained")
    parser.add_argument("--task", required=True, help="Input file name (without .jsonl), file is {data_dir}/{task}.jsonl")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_file = _resolve_data_file(data_dir, args.task)

    summary, no_boxed_rows = analyze_file(data_file, tokenizer_name_or_path=args.model)

    out_data_info = data_dir / "data_info.json"
    out_no_boxed = data_dir / "no_boxed.jsonl"

    # data_info.json: pretty-printed summary for easy viewing.
    write_json(out_data_info, summary)
    write_jsonl(out_no_boxed, no_boxed_rows)

    print(f"data file: {data_file}")
    print(f"wrote: {out_data_info}")
    print(f"wrote: {out_no_boxed}")
    print("bin_counts:")
    for k, v in summary["bin_counts"].items():
        print(f"  {k}: {v}")
    print(f"total_without_boxed: {summary['total_without_boxed']}")
    if summary["invalid_json_lines"]:
        print(f"invalid_json_lines: {summary['invalid_json_lines']}")


if __name__ == "__main__":
    main()
