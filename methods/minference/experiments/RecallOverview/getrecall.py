#!/usr/bin/env python3
"""Evaluate recall results under ./results directory.

Walks the results tree, reads each RecallOverview_*.jsonl file, and computes
recall statistics at multiple granularities (overall, per sample, per layer,
per head), then writes a summary JSON alongside each source file.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass
class OnlineMean:
    total: float = 0.0
    count: int = 0

    def add(self, value: Optional[float]) -> None:
        if value is None:
            return
        self.total += float(value)
        self.count += 1

    def mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.total / self.count


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


# ---------------------------------------------------------------------------
# Iterators over records
# ---------------------------------------------------------------------------

def has_head_weights(records: List[Dict[str, Any]]) -> bool:
    """Return True if any layer in any record carries head_weights."""
    for rec in records:
        recall_list = rec.get("recall")
        if not isinstance(recall_list, list):
            continue
        for layer in recall_list:
            if isinstance(layer, dict) and "head_weights" in layer:
                return True
    return False


def iter_layer_head_metrics(
    record: Dict[str, Any],
) -> Iterator[Tuple[int, int, float, Optional[float]]]:
    """Yield (layer_idx, head_idx, recall, weight|None) for every head."""
    recall_list = record.get("recall")
    if not isinstance(recall_list, list):
        return
    for layer in recall_list:
        if not isinstance(layer, dict):
            continue
        layer_idx = layer.get("layer_idx")
        if not isinstance(layer_idx, int):
            continue
        avg_head_recall = layer.get("avg_head_recall")
        avg_head_weight = layer.get("avg_head_weight")
        yield (float(avg_head_recall), float(avg_head_weight))




# ---------------------------------------------------------------------------
# Granularity 1 – overall average (all rows × all layers × all heads)
# ---------------------------------------------------------------------------

def recall_overall(records: List[Dict[str, Any]], include_weight: bool) -> Dict[str, Any]:
    recall_mean = OnlineMean()
    weight_mean = OnlineMean()
    for rec in records:
        for r, w in iter_layer_head_metrics(rec):
            recall_mean.add(r)
            if include_weight:
                weight_mean.add(w)
    out: Dict[str, Any] = {
        "recall": recall_mean.mean(),
        "count": recall_mean.count,
    }
    if include_weight:
        out["attn_weight"] = weight_mean.mean()
        out["attn_count"] = weight_mean.count
    return out


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_jsonl_files(results_dir: Path) -> List[Path]:
    """Find all RecallOverview_*.jsonl files under results_dir."""
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    files = sorted(results_dir.rglob("RecallOverview_*.jsonl"))
    return files


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate recall results from RecallOverview_*.jsonl files"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./results",
        help="Root directory containing RecallOverview_*.jsonl files (default: ./results)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to write output JSON files (default: same dir as each input)"
    )
    parser.add_argument(
        "--granularity", type=str, choices=["all", "overall", "per_sample", "per_sample_layer", "per_layer", "per_head"],
        default="all",
        help="Which granularities to compute (default: all)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    jsonl_files = discover_jsonl_files(results_dir)
    if not jsonl_files:
        print(f"No RecallOverview_*.jsonl files found under {results_dir}")
        return

    print(f"Found {len(jsonl_files)} file(s):")
    for f in jsonl_files:
        print(f"  {f}")

    for input_path in jsonl_files:
        print(f"\n{'='*60}")
        print(f"Processing: {input_path}")
        records = read_jsonl(input_path)
        if not records:
            print(f"  Skipping empty file: {input_path}")
            continue

        include_weight = has_head_weights(records)
        print(f"  records: {len(records)}, has_attn_weight: {include_weight}")

        granularities: Dict[str, Any] = {}

        if args.granularity in ("all", "overall"):
            granularities["overall"] = recall_overall(records, include_weight)

        out: Dict[str, Any] = {
            "meta": {
                "input_path": str(input_path),
                "num_records": len(records),
                "has_attn_weight": include_weight,
            },
            **granularities,
        }

        # Determine output path
        if output_dir:
            out_dir = output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            # Preserve relative path structure under results_dir
            rel = input_path.relative_to(results_dir)
            out_path = out_dir / rel.parent / f"{input_path.stem}_eval1.json"
        else:
            out_path = input_path.parent / f"{input_path.stem}_eval1.json"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {out_path}")

        # Print a quick summary
        if "overall" in granularities:
            ov = granularities["overall"]
            print(f"  overall recall: {ov['recall']:.6f}" + (
                f"  attn_weight: {ov['attn_weight']:.6f}" if "attn_weight" in ov else ""
            ))

    print(f"\nDone. Processed {len(jsonl_files)} file(s).")


if __name__ == "__main__":
    main()
