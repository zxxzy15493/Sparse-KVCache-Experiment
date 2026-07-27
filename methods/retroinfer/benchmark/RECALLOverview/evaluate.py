#!/usr/bin/env python3

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

TASKS = {
    'niah': 128,
    'vt': 30,
    'cwe': 120,
    'fwe': 50,
    'qa': 32
}

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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
	records: List[Dict[str, Any]] = []
	with path.open("r", encoding="utf-8", errors="ignore") as f:
		for line_no, line in enumerate(f, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				# Skip malformed lines instead of failing the whole run.
				continue
			if isinstance(obj, dict):
				records.append(obj)
	return records


def get_decode_steps(record: Dict[str, Any]) -> List[Dict[str, Any]]:
	steps = record.get("decodeStepList")
	if isinstance(steps, list):
		return [s for s in steps if isinstance(s, dict)]
	return []


def _build_attn_head_map(step: Dict[str, Any]) -> Dict[Tuple[int, int], float]:
	attn_map: Dict[Tuple[int, int], float] = {}
	layer_attn = step.get("layer_attn_weight")
	if not isinstance(layer_attn, list):
		return attn_map

	for item in layer_attn:
		if not isinstance(item, dict):
			continue
		layer_id = item.get("layer_id")
		if not isinstance(layer_id, int):
			continue
		head_list = item.get("head_attn_weight")
		if isinstance(head_list, list):
			for head_idx, v in enumerate(head_list):
				if v is None:
					continue
				attn_map[(layer_id, head_idx)] = float(v)
	return attn_map


def iter_layer_head_metrics(
	record: Dict[str, Any],
) -> Iterator[Tuple[int, int, float, Optional[float]]]:
	"""Yield (layer_id, head_idx, recall, attn_weight?).

	- recall is taken from step['layer_recall'][*]['head_recall'].
	- attn_weight is optional and taken from step['layer_attn_weight'][*]['head_attn_weight'].
	"""

	for step in get_decode_steps(record):
		attn_map = _build_attn_head_map(step)

		layer_recall = step.get("layer_recall")
		if not isinstance(layer_recall, list):
			continue

		for item in layer_recall:
			if not isinstance(item, dict):
				continue
			layer_id = item.get("layer_id")
			if not isinstance(layer_id, int):
				continue
			head_list = item.get("head_recall")
			if not isinstance(head_list, list):
				continue
			for head_idx, r in enumerate(head_list):
				if r is None:
					continue
				attn = attn_map.get((layer_id, head_idx))
				yield (layer_id, head_idx, float(r), attn)


def iter_step_metrics(
	record: Dict[str, Any],
) -> Iterator[Tuple[int, OnlineMean, OnlineMean]]:
	"""Yield per step: (step_id, recall_mean, attn_mean)."""
	for step in get_decode_steps(record):
		step_id = step.get("step")
		if not isinstance(step_id, int):
			# fall back to list index if missing
			step_id = -1
		attn_map = _build_attn_head_map(step)

		recall_mean = OnlineMean()
		attn_mean = OnlineMean()
		layer_recall = step.get("layer_recall")
		if not isinstance(layer_recall, list):
			yield (step_id, recall_mean, attn_mean)
			continue

		for item in layer_recall:
			if not isinstance(item, dict):
				continue
			layer_id = item.get("layer_id")
			if not isinstance(layer_id, int):
				continue
			head_list = item.get("head_recall")
			if not isinstance(head_list, list):
				continue
			for head_idx, r in enumerate(head_list):
				if r is None:
					continue
				recall_mean.add(r)
				attn_mean.add(attn_map.get((layer_id, head_idx)))

		yield (step_id, recall_mean, attn_mean)


def iter_step_layer_metrics(
	record: Dict[str, Any],
) -> Iterator[Tuple[int, int, OnlineMean, OnlineMean]]:
	"""Yield per (step, layer): (step_id, layer_id, recall_mean, attn_mean)."""
	for step in get_decode_steps(record):
		step_id = step.get("step")
		if not isinstance(step_id, int):
			step_id = -1
		attn_map = _build_attn_head_map(step)

		layer_recall = step.get("layer_recall")
		if not isinstance(layer_recall, list):
			continue

		for item in layer_recall:
			if not isinstance(item, dict):
				continue
			layer_id = item.get("layer_id")
			if not isinstance(layer_id, int):
				continue
			head_list = item.get("head_recall")
			if not isinstance(head_list, list):
				continue

			recall_mean = OnlineMean()
			attn_mean = OnlineMean()
			for head_idx, r in enumerate(head_list):
				if r is None:
					continue
				recall_mean.add(r)
				attn_mean.add(attn_map.get((layer_id, head_idx)))
			yield (step_id, layer_id, recall_mean, attn_mean)


def has_attn_weight(records: List[Dict[str, Any]]) -> bool:
	for rec in records:
		for step in get_decode_steps(rec):
			if "layer_attn_weight" in step:
				return True
	return False


def recall_1(records: List[Dict[str, Any]], include_attn: bool) -> Dict[str, Any]:
	"""All rows + all steps + all layers + all heads average."""
	recall_mean = OnlineMean()
	attn_mean = OnlineMean()
	for rec in records:
		for _, _, r, a in iter_layer_head_metrics(rec):
			recall_mean.add(r)
			if include_attn:
				attn_mean.add(a)
	out: Dict[str, Any] = {
		"recall": recall_mean.mean(),
		"count": recall_mean.count,
	}
	if include_attn:
		out["attn_weight"] = attn_mean.mean()
		out["attn_count"] = attn_mean.count
	return out


def recall_2(records: List[Dict[str, Any]], include_attn: bool) -> List[Dict[str, Any]]:
	"""Per row average across all steps/layers/heads."""
	result: List[Dict[str, Any]] = []
	for row_idx, rec in enumerate(records):
		recall_mean = OnlineMean()
		attn_mean = OnlineMean()
		for _, _, r, a in iter_layer_head_metrics(rec):
			recall_mean.add(r)
			if include_attn:
				attn_mean.add(a)

		row_out: Dict[str, Any] = {
			"row": row_idx,
			"index": rec.get("index", row_idx),
			"recall": recall_mean.mean(),
			"count": recall_mean.count,
		}
		if include_attn:
			row_out["attn_weight"] = attn_mean.mean()
			row_out["attn_count"] = attn_mean.count
		result.append(row_out)
	return result


def recall_3(records: List[Dict[str, Any]], include_attn: bool) -> List[Dict[str, Any]]:
	"""Per row, per decode step average across all layers/heads."""
	result: List[Dict[str, Any]] = []
	for row_idx, rec in enumerate(records):
		steps_out: List[Dict[str, Any]] = []
		for step_id, recall_mean, attn_mean in iter_step_metrics(rec):
			step_out: Dict[str, Any] = {
				"step": step_id,
				"recall": recall_mean.mean(),
				"count": recall_mean.count,
			}
			if include_attn:
				step_out["attn_weight"] = attn_mean.mean()
				step_out["attn_count"] = attn_mean.count
			steps_out.append(step_out)

		row_out: Dict[str, Any] = {
			"row": row_idx,
			"index": rec.get("index", row_idx),
			"steps": steps_out,
		}
		result.append(row_out)
	return result


def recall_4(records: List[Dict[str, Any]], include_attn: bool) -> List[Dict[str, Any]]:
	"""Per row, per step, per layer average across heads."""
	result: List[Dict[str, Any]] = []
	for row_idx, rec in enumerate(records):
		# build step->layers list
		step_to_layers: Dict[int, List[Dict[str, Any]]] = {}
		for step_id, layer_id, recall_mean, attn_mean in iter_step_layer_metrics(rec):
			layer_out: Dict[str, Any] = {
				"layer_id": layer_id,
				"recall": recall_mean.mean(),
				"count": recall_mean.count,
			}
			if include_attn:
				layer_out["attn_weight"] = attn_mean.mean()
				layer_out["attn_count"] = attn_mean.count

			step_to_layers.setdefault(step_id, []).append(layer_out)

		steps_out = [
			{"step": step_id, "layers": sorted(layers, key=lambda x: x["layer_id"])}
			for step_id, layers in sorted(step_to_layers.items(), key=lambda x: x[0])
		]
		result.append(
			{
				"row": row_idx,
				"index": rec.get("index", row_idx),
				"steps": steps_out,
			}
		)
	return result


def recall_5(records: List[Dict[str, Any]], include_attn: bool) -> Dict[str, Any]:
	"""All rows + all steps, per layer average across heads."""
	layer_recall: Dict[int, OnlineMean] = {}
	layer_attn: Dict[int, OnlineMean] = {}
	for rec in records:
		for layer_id, _, r, a in iter_layer_head_metrics(rec):
			layer_recall.setdefault(layer_id, OnlineMean()).add(r)
			if include_attn:
				layer_attn.setdefault(layer_id, OnlineMean()).add(a)

	layers_out: List[Dict[str, Any]] = []
	for layer_id in sorted(layer_recall.keys()):
		out: Dict[str, Any] = {
			"layer_id": layer_id,
			"recall": layer_recall[layer_id].mean(),
			"count": layer_recall[layer_id].count,
		}
		if include_attn:
			out["attn_weight"] = layer_attn.get(layer_id, OnlineMean()).mean()
			out["attn_count"] = layer_attn.get(layer_id, OnlineMean()).count
		layers_out.append(out)

	result: Dict[str, Any] = {"layers": layers_out}
	return result

def recall_6(records: List[Dict[str, Any]], include_attn: bool) -> Dict[str, Any]:
	"""All rows + all steps, per (layer, head) average."""
	recall_map: Dict[Tuple[int, int], OnlineMean] = {}
	attn_map: Dict[Tuple[int, int], OnlineMean] = {}
	for rec in records:
		for layer_id, head_idx, r, a in iter_layer_head_metrics(rec):
			recall_map.setdefault((layer_id, head_idx), OnlineMean()).add(r)
			if include_attn:
				attn_map.setdefault((layer_id, head_idx), OnlineMean()).add(a)

	# group by layer
	layer_to_heads: Dict[int, Dict[int, Dict[str, Any]]] = {}
	for (layer_id, head_idx), mean_obj in recall_map.items():
		layer_entry = layer_to_heads.setdefault(layer_id, {})
		head_out: Dict[str, Any] = {
			"head": head_idx,
			"recall": mean_obj.mean(),
			"count": mean_obj.count,
		}
		if include_attn:
			aobj = attn_map.get((layer_id, head_idx), OnlineMean())
			head_out["attn_weight"] = aobj.mean()
			head_out["attn_count"] = aobj.count
		layer_entry[head_idx] = head_out

	layers_out: List[Dict[str, Any]] = []
	for layer_id in sorted(layer_to_heads.keys()):
		heads = [layer_to_heads[layer_id][h] for h in sorted(layer_to_heads[layer_id].keys())]
		layer_out: Dict[str, Any] = {
			"layer_id": layer_id,
			"heads": heads,
		}
		layers_out.append(layer_out)

	return {"layers": layers_out}

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Analyze decodeStepList recall/attn_weight from a jsonl file")
	parser.add_argument("--data_dir", type=str, help="Directory containing the data file")
	parser.add_argument("--task", type=str, help="Data filename stem (input is {data_dir}/{task}.jsonl)")
	parser.add_argument("--budget", type=int, help="Budget for filtering steps/layers/heads (if applicable)")
	return parser.parse_args()

def main() -> None:
	args = parse_args()
	budget = args.budget
	if args.task in ['narrativeqa', 'qasper']:
		task_key = args.task
	else:
		task_key = [key for key in TASKS.keys() if key in args.task][0]
	data_dir = Path(args.data_dir + f"/{task_key}")
	tasks = [f"RECALLOverview_{args.task}_top100.jsonl", f"RECALLOverview_{args.task}_{budget}.jsonl"]
	for task in tasks:
		input_path = data_dir / (task if task.endswith(".jsonl") else f"{task}.jsonl")
		if not input_path.exists():
			raise FileNotFoundError(f"Input file not found: {input_path}")

		records = read_jsonl(input_path)
		include_attn = has_attn_weight(records)

		out: Dict[str, Any] = {
			"meta": {
				"data_dir": str(data_dir),
				"task": task,
				"input_path": str(input_path),
				"num_rows": len(records),
				"has_attn_weight": include_attn,
			},
			"recall_1": recall_1(records, include_attn),
			"recall_2": recall_2(records, include_attn),
			"recall_3": recall_3(records, include_attn),
			"recall_4": recall_4(records, include_attn),
			"recall_5": recall_5(records, include_attn),
			"recall_6": recall_6(records, include_attn),
		}
		if 'top100' in task:
			result_path = data_dir / "result_top100.json"
		else:
			result_path = data_dir / "result.json"
		result_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
		print(f"Wrote result: {result_path}")

if __name__ == "__main__":
	main()
