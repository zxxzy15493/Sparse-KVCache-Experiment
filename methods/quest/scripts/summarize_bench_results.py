import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


FILENAME_RE = re.compile(
  r"^(?P<model>[^-]+)-budget(?P<budget>\d+)-chunk_size\d+-seqlen(?P<seqlen>\d+)\.jsonl$"
)


def iter_run_averages(path):
  current = []

  def flush():
    if not current:
      return None
    selected = current[2:4]
    if len(selected) != 2:
      raise ValueError(f"{path}: expected at least 4 iterations in a run, got {len(current)}")
    return {
      "ttft_s": mean(row["ttft_s"] for row in selected),
      "tpot_ms": mean(row["tpot_ms"] for row in selected),
      "total_latency_s": mean(row["total_latency_s"] for row in selected),
    }

  with path.open() as f:
    for line_no, line in enumerate(f, 1):
      row = json.loads(line)
      if "bench_type" in row:
        avg = flush()
        if avg is not None:
          yield avg
        current = []
        continue

      if "iteration" not in row:
        raise ValueError(f"{path}:{line_no}: iteration row missing iteration field")
      current.append(row)

  avg = flush()
  if avg is not None:
    yield avg


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "results_dir",
    nargs="?",
    default=Path(__file__).with_name("bench_results"),
    type=Path,
  )
  parser.add_argument(
    "-o",
    "--output",
    default=None,
    type=Path,
    help="Output CSV path. Defaults to <results_dir>/efficiency_summary.csv.",
  )
  args = parser.parse_args()

  results_dir = args.results_dir
  output = args.output or results_dir / "efficiency_summary.csv"

  grouped = defaultdict(list)
  budgets = set()

  for path in sorted(results_dir.glob("*.jsonl")):
    match = FILENAME_RE.match(path.name)
    if not match:
      continue

    model = match.group("model")
    budget = int(match.group("budget"))
    seqlen = int(match.group("seqlen"))
    run_averages = list(iter_run_averages(path))
    if not run_averages:
      continue

    budgets.add(budget)
    grouped[(model, seqlen, budget)].extend(run_averages)

  sorted_budgets = sorted(budgets)
  header = ["model", "input_length"]
  for budget in sorted_budgets:
    header.extend(
      [
        f"budget{budget}_ttft",
        f"budget{budget}_tpot",
        f"budget{budget}_latency",
      ]
    )

  rows = []
  model_seqlens = sorted({(model, seqlen) for model, seqlen, _ in grouped})
  for model, seqlen in model_seqlens:
    row = {"model": model, "input_length": seqlen}
    for budget in sorted_budgets:
      values = grouped.get((model, seqlen, budget))
      if not values:
        continue
      row[f"budget{budget}_ttft"] = f"{mean(v['ttft'] for v in values):.2f}"
      row[f"budget{budget}_tpot"] = f"{mean(v['tpot'] for v in values):.2f}"
      row[f"budget{budget}_latency"] = f"{mean(v['latency'] for v in values):.2f}"
    rows.append(row)

  with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)

  print(f"Wrote {output}")


if __name__ == "__main__":
  main()
