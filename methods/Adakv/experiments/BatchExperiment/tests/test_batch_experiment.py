"""Minimal checks for batch splitting, resume filtering and padded-width slicing.

These tests avoid any GPU / model download: they only exercise the pure helpers
and ``get_pred`` against a tiny in-memory fake model on CPU.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))
spec = importlib.util.spec_from_file_location("adakv_batch_pred", EXPERIMENT_ROOT / "pred.py")
pred = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pred)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class Tokenizer:
    pad_token_id = 99
    eos_token_id = 99
    padding_side = "right"

    def __call__(self, prompts, **kwargs):
        if self.padding_side != "left":
            raise AssertionError("Decoder-only batching must use left padding")
        rows = [[int(token) for token in prompt.split()] for prompt in prompts]
        width = max(map(len, rows))
        ids = [[99] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return SimpleNamespace(input_ids=torch.tensor(ids), attention_mask=torch.tensor(masks))

    def batch_decode(self, rows, **kwargs):
        return [" ".join(str(token) for token in row if token != 99) for row in rows]


class BatchExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def args(self, *extra):
        return pred.parse_args([
            "--model", "qwen", "--benchmark", "ruler", "--task", "niah_single_1",
            "--save_dir", str(self.root / "results"), "--context_length", "4096", *extra,
        ])

    def test_benchmark_aliases_and_parameter_validation(self):
        for name in ("RULER", "ruler", "Synthetic", "synthetic"):
            with self.subTest(name=name):
                args = self.args("--benchmark", name, "--batch_size", "3")
                self.assertEqual((args.benchmark, args.context_length, args.batch_size), ("RULER", 4096, 3))
                self.assertEqual(args.model, "Qwen/Qwen2.5-7B-Instruct-1M")
        for extra in (["--batch_size", "0"], ["--batch_size", "-1"], ["--num_samples", "0"]):
            with self.subTest(extra=extra):
                with self.assertRaises(SystemExit):
                    self.args(*extra)

    def test_iter_batches_splits_evenly_and_keeps_tail(self):
        self.assertEqual([len(b) for b in pred.iter_batches(list(range(5)), 2)], [2, 2, 1])
        self.assertEqual([b for b in pred.iter_batches([10, 11, 12], 4)], [[10, 11, 12]])

    def test_resume_filter_applies_num_samples_then_skips_completed(self):
        task_file = self.root / "validation.jsonl"
        write_jsonl(task_file, [{"index": index, "input": f"p {index}", "outputs": [str(index)]}
                                for index in range(5)])
        args = self.args("--data_dir", str(task_file), "--batch_size", "2", "--num_samples", "4")
        write_jsonl(args.save_dir / "niah_single_1.jsonl", [{"index": 0, "pred": "0"}])
        remaining, max_new_tokens, input_budget, _ = pred.load_dataset(args)
        self.assertEqual([record["index"] for record in remaining], [1, 2, 3])
        self.assertEqual([len(batch) for batch in pred.iter_batches(remaining, 2)], [2, 1])
        self.assertEqual(max_new_tokens, 128)
        self.assertIsNone(input_budget)

    def test_duplicate_ids_fail(self):
        task_file = self.root / "validation.jsonl"
        write_jsonl(task_file, [{"index": "same", "input": "1", "outputs": ["ok"]}] * 2)
        args = self.args("--data_dir", str(task_file))
        with self.assertRaisesRegex(ValueError, "Duplicate sample index"):
            pred.load_dataset(args)

    def test_generate_slices_at_padded_width_and_trims_each_eos(self):
        samples = [{"index": i, "input": "1 2 " + "3 " * i,
                    "answers": [str(i)], "outputs": [str(i)]} for i in range(3)]
        calls = []
        tails = iter([[[10, 98, 99], [20, 21, 99]], [[30, 99]]])

        def generate(**kwargs):
            calls.append(kwargs)
            return torch.cat([kwargs["input_ids"], torch.tensor(next(tails))], dim=1)

        model = SimpleNamespace(device=torch.device("cpu"), generate=generate,
                                generation_config=SimpleNamespace(eos_token_id=[98, 99]))
        out_path = self.root / "LongBench" / "narrativeqa.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for batch in pred.iter_batches(samples, 2):
            pred.get_pred(model, Tokenizer(), batch, 128, "LongBench", out_path, None, 2)
        self.assertEqual([call["input_ids"].shape[0] for call in calls], [2, 1])
        self.assertEqual(calls[0]["input_ids"].tolist(), [[99, 1, 2], [1, 2, 3]])
        self.assertEqual(calls[0]["attention_mask"].tolist(), [[0, 1, 1], [1, 1, 1]])
        records = pred.load_data(out_path)
        self.assertEqual([(r["index"], r["pred"], r["output_tokens"]) for r in records],
                         [(0, "10", 1), (1, "20 21", 2), (2, "30", 1)])
        stats = pred.load_data(out_path.with_suffix(".batch_stats.jsonl"))
        self.assertEqual([r["batch_size"] for r in stats], [2, 1])
        self.assertEqual([r["configured_batch_size"] for r in stats], [2, 2])
        self.assertTrue(all(r["latency_s"] > 0 for r in stats))


if __name__ == "__main__":
    unittest.main()
