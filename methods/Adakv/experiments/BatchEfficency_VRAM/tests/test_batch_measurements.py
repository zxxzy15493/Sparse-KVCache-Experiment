"""Metric correctness and manifest/CRUD checks for the shared batch measurement module.

Pure-CPU unit tests: no CUDA and no model checkpoint download.
"""

import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import pred  # noqa: E402


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                    encoding="utf-8")


class Tokenizer:
    """A left-padding tokenizer stub matching how pred.tokenize_batch drives tokenizers."""

    pad_token_id = 99
    eos_token_id = 99
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"
    truncation_side = "left"

    def __call__(self, prompts, **kwargs):
        rows = [[int(token) for token in prompt.split()] for prompt in prompts]
        return {"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows]}

    def pad(self, data, padding=True, return_tensors="pt", **kwargs):
        rows = data["input_ids"]
        width = max(map(len, rows))
        input_ids = torch.tensor([[99] * (width - len(row)) + row for row in rows])
        attention_mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, tokens, skip_special_tokens=True):
        return [" ".join(str(int(token)) for token in row) for row in tokens.tolist()]


def summary_file(path, suffix):
    """Path to the sibling .summary.txt/.summary.csv of a returned .summary.json path."""
    return Path(str(path)[:-len(".summary.json")] + suffix)


def ruler_args(root, measurement="efficency", data_file=None, batch_size="2", **extra):
    argv = ["--model", "llama-3.1-8b", "--benchmark", "ruler", "--task", "niah_single_1",
            "--save_dir", str(root / "results"), "--batch_size", batch_size,
            "--fixed_output_length", "4", "--warmup", "0"]
    if data_file is not None:
        argv += ["--data_dir", str(data_file)]
    for key, value in extra.items():
        argv += [f"--{key}", str(value)]
    return argv


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def samples(self):
        return [{"input": "1 " + "2 " * (i + 1)} for i in range(3)]

    def data_file(self, records=None):
        path = self.root / "validation.jsonl"
        write_jsonl(path, self.samples() if records is None else records)
        return path

    # ---- manifest / resume / consistency (no model, no CUDA) ----

    def test_manifest_written_and_resume_filters_completed_batches(self):
        path = self.data_file()
        args = pred.parse_args(ruler_args(self.root, data_file=path), measurement="efficency")
        remaining, max_new_tokens, _ = pred.load_dataset(args)
        self.assertEqual([s["index"] for s in remaining], [0, 1, 2])
        self.assertEqual(max_new_tokens, 128)
        manifest = self.root / "results/niah_single_1.efficency.run.json"
        self.assertTrue(manifest.is_file())
        data = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], pred.SCHEMA_VERSION)
        self.assertEqual(data["loader_fingerprint"], pred.LOADER_FINGERPRINT)
        self.assertTrue(data["input_sha256"])
        # a complete batch is now saved -> resume should leave only the tail
        write_jsonl(self.root / "results/niah_single_1.efficency.jsonl",
                    [{"schema_version": 1, "indices": ["0", "1"], "batch_size": 2}])
        remaining, _, _ = pred.load_dataset(args)
        self.assertEqual([s["index"] for s in remaining], [2])

    def test_manifest_rejects_changed_configuration(self):
        path = self.data_file()
        args = pred.parse_args(ruler_args(self.root, data_file=path), measurement="efficency")
        pred.load_dataset(args)
        changed = pred.parse_args(ruler_args(self.root, data_file=path, batch_size="1"),
                                  measurement="efficency")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Run configuration or input changed"):
            pred.load_dataset(changed)

    def test_manifest_rejects_changed_input(self):
        path = self.data_file()
        args = pred.parse_args(ruler_args(self.root, data_file=path), measurement="efficency")
        with redirect_stdout(io.StringIO()):
            pred.load_dataset(args)
        path.write_text("".join(json.dumps({"input": "3 4"}) + "\n" for _ in range(3)), encoding="utf-8")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "input_sha256"):
            pred.load_dataset(args)

    def test_existing_metrics_without_manifest_are_rejected(self):
        path = self.data_file()
        metrics = self.root / "results/niah_single_1.efficency.jsonl"
        write_jsonl(metrics, [{"schema_version": 1, "indices": ["0"], "batch_size": 1}])
        args = pred.parse_args(ruler_args(self.root, data_file=path), measurement="efficency")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "no run manifest"):
            pred.load_dataset(args)

    def test_invalid_or_partial_batches_are_rejected(self):
        path = self.data_file()
        args = pred.parse_args(ruler_args(self.root, data_file=path), measurement="efficency")
        with redirect_stdout(io.StringIO()):
            pred.load_dataset(args)
        # indices (0, 2) are not a planned batch for batch_size=2
        bad = self.root / "results/niah_single_1.efficency.jsonl"
        write_jsonl(bad, [{"schema_version": 1, "indices": ["0", "2"], "batch_size": 2}])
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Invalid or duplicated"):
            pred.load_dataset(args)

    # ---- batching, padding slicing ----

    def test_iter_batches_keeps_tail(self):
        self.assertEqual([list(pred.iter_batches([1, 2, 3, 4, 5], 2))],
                         [[[1, 2], [3, 4], [5]]])

    def test_tokenize_batch_pads_left_and_truncates_long_prompts(self):
        tokenizer = Tokenizer()
        samples = [{"input": "1 2 3"}, {"input": "1 2"}]
        inputs = pred.tokenize_batch(tokenizer, samples, max_len=3)
        self.assertEqual(inputs["input_ids"].tolist(), [[1, 2, 3], [99, 1, 2]])
        self.assertEqual(inputs["attention_mask"].tolist(), [[1, 1, 1], [0, 1, 1]])
        long_prompt = [{"input": " ".join(str(i) for i in range(10))}]
        sliced = pred.tokenize_batch(tokenizer, long_prompt, max_len=4)
        self.assertEqual(sliced["input_ids"].shape[1], 4)

    # ---- derived throughput fields ----

    def test_efficiency_derived_backfills_old_records(self):
        records = [{"batch_size": 2, "ttft": 1.0, "decode_steps": 3, "decode_latency": 0.06,
                    "latency": 1.3, "input_tokens": [3, 4], "output_tokens": [4, 4]}]
        path = self.root / "old.jsonl"
        write_jsonl(path, records)
        loaded = pred.load_efficiency_records(path)
        self.assertAlmostEqual(loaded[0]["throughput"], 2 * 3 / 0.06)
        self.assertAlmostEqual(loaded[0]["prefill_tokens_per_s"], 7 / 1.0)
        self.assertAlmostEqual(loaded[0]["decode_tokens_per_s"], 6 / 0.06)
        self.assertAlmostEqual(loaded[0]["output_tokens_per_s"], 8 / 1.3)

    # ---- summary csv/txt, weighted TPOT ----

    def test_efficiency_summary_weights_tpot_and_writes_csv(self):
        records = [
            {"batch_size": 2, "ttft": 1.0, "TPOT": 5.0, "decode_steps": 2, "decode_latency": 0.02,
             "latency": 1.5, "input_tokens": [2, 3], "output_tokens": [3, 3],
             "throughput": 200.0},
            {"batch_size": 2, "ttft": 3.0, "TPOT": 7.0, "decode_steps": 4, "decode_latency": 0.08,
             "latency": 3.5, "input_tokens": [4, 5], "output_tokens": [5, 5],
             "throughput": 100.0},
        ]
        metrics = self.root / "results/niah_single_1.efficency.jsonl"
        write_jsonl(metrics, records)
        args = pred.parse_args(ruler_args(self.root, data_file=None), measurement="efficency")
        with redirect_stdout(io.StringIO()):
            path = pred.write_summary(args)
        summary = json.loads(path.read_text(encoding="utf-8"))
        values = summary["by_batch_size"]["2"]
        self.assertAlmostEqual(values["mean_ttft_s"], 2.0)
        self.assertAlmostEqual(values["mean_tpot_ms"], 6.0)
        self.assertAlmostEqual(values["weighted_tpot_ms"], 0.10 * 1000 / 6)
        self.assertAlmostEqual(values["throughput_tokens_per_s"], 120.0)
        self.assertAlmostEqual(values["prefill_tokens_per_s"], 14 / 4)
        self.assertAlmostEqual(values["decode_tokens_per_s"], 120.0)
        self.assertTrue(summary_file(path, ".summary.txt").is_file())
        with summary_file(path, ".summary.csv").open() as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["batch"], "2")
        self.assertAlmostEqual(float(row["weightedTPOT_ms"]), 100 / 6)
        self.assertAlmostEqual(float(row["decode_tokens_per_s"]), 120.0)

    def test_vram_summary_expands_per_device_columns(self):
        records = [{
            "batch_size": 2, "peak_alloc": 100.0, "peak_reserved": 200.0,
            "peak_alloc_by_device": {"cuda:0": 100.0, "cuda:1": 50.0},
            "peak_reserved_by_device": {"cuda:0": 200.0, "cuda:1": 90.0},
            "decode_num_steps": 3, "decode_avg_alloc": 20.0, "decode_peak_alloc": 30.0,
            "decode_avg_alloc_by_device": {"cuda:0": 20.0, "cuda:1": 15.0},
            "decode_peak_alloc_by_device": {"cuda:0": 30.0, "cuda:1": 25.0},
        }]
        metrics = self.root / "results/niah_single_1.vram.jsonl"
        write_jsonl(metrics, records)
        args = pred.parse_args(ruler_args(self.root, data_file=None), measurement="vram")
        with redirect_stdout(io.StringIO()):
            path = pred.write_summary(args)
        summary = json.loads(path.read_text(encoding="utf-8"))
        values = summary["by_batch_size"]["2"]
        self.assertEqual(values["peak_alloc_mib"], 100.0)
        self.assertEqual(values["decode_avg_alloc_mib"], 20.0)
        text = summary_file(path, ".summary.txt").read_text(encoding="utf-8")
        self.assertIn("cuda:0", text)
        with summary_file(path, ".summary.csv").open() as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["cuda:0_peak_alloc_mib"], "100.0")

    # ---- device discovery (constructing torch.device needs no physical GPU) ----

    def test_model_cuda_device_discovery_filters_cpu_and_disk(self):
        cpu_model = SimpleNamespace(device="cpu", hf_device_map={},
                                    parameters=lambda: iter([]), buffers=lambda: iter([]))
        self.assertEqual(pred.model_cuda_devices(cpu_model), [])
        multi = SimpleNamespace(device="cuda:0", hf_device_map={"layer": 1, "head": "disk"},
                                parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda:2"))]),
                                buffers=lambda: iter([]))
        self.assertEqual(list(map(str, pred.model_cuda_devices(multi))), ["cuda:0", "cuda:1", "cuda:2"])


if __name__ == "__main__":
    unittest.main()
