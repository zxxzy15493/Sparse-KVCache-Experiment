from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common import method_defaults, output_path, parse_set, run_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate unified LongBenchV2 predictions")
    parser.add_argument("--output-root", default=str(Path(__file__).parent / "outputs"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if args.budget is None:
        args.budget = method_defaults(args.method)["budget"]
    args.run = run_name(args.method, args.budget, parse_set(args.set, args.method))
    return args


def percentage(correct: int, total: int) -> float | None:
    return round(100 * correct / total, 1) if total else None


def evaluate(args: argparse.Namespace) -> Path:
    path = output_path(args.output_root, args.model, args.method, "longbenchv2", args.run)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = {"overall": [], "easy": [], "hard": [], "short": [], "medium": [], "long": []}
    for row in rows:
        groups["overall"].append(bool(row["judge"]))
        groups["easy" if row["difficulty"] == "easy" else "hard"].append(bool(row["judge"]))
        groups[row["length"]].append(bool(row["judge"]))
    result = {
        name: {"accuracy": percentage(sum(values), len(values)), "total": len(values)}
        for name, values in groups.items()
    }
    result_path = path.with_name(f"result-{path.stem}.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result_path


def main(argv: list[str] | None = None) -> None:
    print(evaluate(parse_args(argv)))


if __name__ == "__main__":
    main()
