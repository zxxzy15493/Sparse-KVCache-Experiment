"""Regression checks for prediction and independent measurement components."""

import json
from functools import wraps
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from benchmarks.common import parse_set
from benchmarks.efficency_vram.tools.efficiency import EfficiencyMeter
from benchmarks.efficency_vram.tools.memory import MemoryMonitor
from benchmarks.efficency_vram.pred import (
    generate, iter_batches, model_devices, normalize_tokens, parse_args,
    predict_batch, prepare_batch, resolve_data_file, run,
)
from benchmarks.efficency_vram.tools.reporting import summarize


def tiny_model():
    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, max_position_embeddings=128, eos_token_id=0, pad_token_id=0)
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


class NumericTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [[int(token) for token in text.split()] for text in texts]}

    def batch_decode(self, rows, **kwargs):
        return [" ".join(map(str, row.tolist() if isinstance(row, torch.Tensor) else row)) for row in rows]


@pytest.mark.parametrize("length", [1, 4])
def test_forward_matches_hf_generation_and_uses_cached_decode(length):
    model = tiny_model()
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
    expected = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=length,
                              eos_token_id=None, do_sample=False, pad_token_id=0)
    calls = []
    original = model.forward

    @wraps(original)
    def observed(*args, **kwargs):
        calls.append((kwargs["input_ids"].shape[1], kwargs["cache_position"].tolist()))
        return original(*args, **kwargs)

    model.forward = observed
    with EfficiencyMeter((), 2, length) as timer:
        actual = generate(model, inputs, torch.ones_like(inputs), length, step_context=timer.step)
    metrics = timer.metrics()
    assert torch.equal(actual, expected[:, inputs.shape[1]:])
    assert [size for size, _ in calls] == [3] + [1] * (length - 1)
    assert [positions for _, positions in calls] == [[0, 1, 2]] + [[position] for position in range(3, 3 + length - 1)]
    assert metrics["decode_steps"] == length - 1
    assert len(metrics["step_latencies_s"]) == length
    assert (metrics["tpot_ms"] is None) == (length == 1)


def test_prediction_without_measurements_ignores_eos():
    model = tiny_model()
    with torch.no_grad():
        model.lm_head.weight.zero_()
    inputs = torch.tensor([[1, 2], [3, 4]])
    output = generate(model, inputs, torch.ones_like(inputs), 5)
    assert output.shape == (2, 5)
    assert output.eq(0).all()


def test_efficiency_units_and_step_scope():
    clock = [1.0, 1.1, 1.3, 1.4, 1.41, 1.5, 1.53, 1.8]
    with patch("benchmarks.efficency_vram.tools.efficiency.time.perf_counter", side_effect=clock):
        with EfficiencyMeter((), 2, 3) as timer:
            for _ in range(3):
                with timer.step():
                    pass
    metrics = timer.metrics()
    assert metrics["ttft_s"] == pytest.approx(0.2)
    assert metrics["tpot_ms"] == pytest.approx(20)
    assert metrics["generation_latency_s"] == pytest.approx(0.8)
    assert metrics["decode_throughput_tokens_per_s"] == pytest.approx(100)
    assert metrics["output_throughput_tokens_per_s"] == pytest.approx(7.5)


def test_sharded_memory_baseline_and_peaks():
    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    mib = 1024**2
    with patch("torch.cuda.synchronize") as sync, patch("torch.cuda.reset_peak_memory_stats") as reset, \
         patch("torch.cuda.memory_allocated", side_effect=[10*mib, 20*mib]), \
         patch("torch.cuda.memory_reserved", side_effect=[15*mib, 25*mib]), \
         patch("torch.cuda.max_memory_allocated", side_effect=[30*mib, 50*mib]), \
         patch("torch.cuda.max_memory_reserved", side_effect=[35*mib, 60*mib]):
        with MemoryMonitor(devices) as memory:
            pass
    metrics = memory.metrics()
    assert reset.call_count == 2 and sync.call_count == 4
    assert metrics["peak_allocated_mib"] == 50
    assert metrics["sum_device_peak_allocated_mib"] == 80
    assert metrics["memory_by_device"]["cuda:1"]["incremental_peak_allocated_mib"] == 30


def test_memory_only_does_not_change_generation_or_collect_timing():
    model = tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    expected = generate(model, ids, torch.ones_like(ids), 3)
    with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
         patch("torch.cuda.memory_allocated", return_value=0), patch("torch.cuda.memory_reserved", return_value=0), \
         patch("torch.cuda.max_memory_allocated", return_value=1024), \
         patch("torch.cuda.max_memory_reserved", return_value=2048):
        with MemoryMonitor((torch.device("cuda:0"),)) as memory:
            actual = generate(model, ids, torch.ones_like(ids), 3)
    assert torch.equal(actual, expected)
    assert "ttft_s" not in memory.metrics()


@pytest.mark.parametrize("measurement", ["vram", "both"])
def test_prediction_composes_independent_monitors(measurement):
    args = parse_args(["--model", "qwen3-4b", "--input-length", "4", "--fixed-output-length", "3",
                       "--measurement", measurement])
    batch = [{"index": 0, "input": "1 2 3"}, {"index": 1, "input": "4 5 6"}]
    with patch("torch.cuda.synchronize"), patch("torch.cuda.reset_peak_memory_stats"), \
         patch("torch.cuda.memory_allocated", return_value=0), patch("torch.cuda.memory_reserved", return_value=0), \
         patch("torch.cuda.max_memory_allocated", return_value=1024**2), \
         patch("torch.cuda.max_memory_reserved", return_value=2 * 1024**2), \
         patch("torch.cuda.device"), patch("torch.cuda.empty_cache"):
        tokens, metrics = predict_batch(args, tiny_model(), NumericTokenizer(), batch, "cpu", (torch.device("cuda:0"),))
    assert tokens.shape == (2, 3)
    assert metrics["peak_allocated_mib"] == 1
    assert ("ttft_s" in metrics) == (measurement == "both")
    if measurement == "both":
        assert len(metrics["step_latencies_s"]) == 3


def test_exact_input_length_and_batches():
    rows = [{"index": 1, "input": "1 2 3 4 5 6"}, {"index": 2, "input": "7 8"}]
    ids, mask, metadata = prepare_batch(rows, NumericTokenizer(), 4, "cpu")
    assert ids.tolist() == [[1, 2, 5, 6], [7, 7, 8, 8]]
    assert mask.eq(1).all()
    assert metadata["original_input_tokens"] == [6, 2]
    assert metadata["input_adjustments"] == ["middle_truncated", "repeated"]
    assert normalize_tokens([1, 2, 3], 1) == ([3], "middle_truncated")
    with pytest.raises(ValueError, match="less than requested"):
        normalize_tokens([1, 2], 4, "error")
    assert [len(batch) for batch in iter_batches(rows * 2 + rows[:1], 2)] == [2, 2]
    assert [len(batch) for batch in iter_batches(rows * 2 + rows[:1], 2, True)] == [2, 2, 1]


def test_cli_translates_loader_options():
    args = parse_args(["--method", "streaming", "--sink_token", "8", "--local_window_token", "56"])
    assert args.budget == 64
    assert parse_set(args.set) == {"start_size": 8, "recent_size": 56}
    assert parse_args(["--method", "minference"]).budget == 1024
    args = parse_args(["--method", "retroinfer", "--set", "estimate_ratio=0.5"])
    assert args.estimate_ratio == 0.5
    args = parse_args(["--method", "pqcache", "--input_length", "8192", "--fixed_output_length", "2048"])
    assert args.max_seq_length + 1024 >= args.input_length + args.fixed_output_length


@pytest.mark.parametrize("method", ["retroinfer", "magicpig"])
def test_native_fixed_length_and_total_timing(method):
    class Model:
        fixed_output_length = 0

        def generate(self, **kwargs):
            assert (kwargs.get("fixed_output_length") if method == "retroinfer" else self.fixed_output_length) == 3
            return torch.cat([kwargs["input_ids"], torch.ones((1, 3), dtype=torch.long)], dim=1)

    model = Model()
    ids = torch.tensor([[4, 5]])
    with EfficiencyMeter((), 1, 3) as timer:
        tokens = generate(model, ids, torch.ones_like(ids), 3, method, timer.step)
    assert tokens.shape == (1, 3)
    assert model.fixed_output_length == 0
    assert timer.metrics()["generation_latency_s"] > 0
    assert timer.metrics()["ttft_s"] is None and timer.metrics()["tpot_ms"] is None


def test_native_magicpig_restores_state_after_failure():
    model = SimpleNamespace(fixed_output_length=7, generate=Mock(side_effect=RuntimeError("failure")))
    with pytest.raises(RuntimeError, match="failure"):
        generate(model, torch.ones((1, 2)), None, 3, "magicpig")
    assert model.fixed_output_length == 7


def test_device_discovery_uses_model_placement():
    weight = SimpleNamespace(device=torch.device("cuda:0"))
    model = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=weight),
                            parameters=lambda: [weight, SimpleNamespace(device=torch.device("cuda:1")),
                                                SimpleNamespace(device=torch.device("cpu"))])
    assert model_devices(model, "full_attention") == (torch.device("cuda:0"), (torch.device("cuda:0"), torch.device("cuda:1")))
    native = SimpleNamespace(layers=[SimpleNamespace(device="cuda:1"), SimpleNamespace(device="cuda:0")])
    assert model_devices(native, "retroinfer") == (torch.device("cuda:1"), (torch.device("cuda:0"), torch.device("cuda:1")))


def test_entrypoint_warmup_batches_reporting_and_overwrite(tmp_path):
    data_file = tmp_path / "validation.jsonl"
    data_file.write_text("\n".join(json.dumps({"index": index, "input": "1 2 3"}) for index in range(5)))
    args = parse_args(["--model", "qwen3-4b", "--method", "streaming", "--measurement", "efficiency", "--device", "cpu",
                       "--data-dir", str(data_file), "--output-root", str(tmp_path / "output"),
                       "--batch_size", "2", "--input_length", "4", "--fixed_output_length", "2",
                       "--num-samples", "5", "--allow-partial-batch", "--warmup", "1", "--save-predictions"])
    cleanup = Mock()

    def load(options):
        options.model_path = "local-tiny-qwen3"
        options._method_args = SimpleNamespace(sink_size=16)
        options._cleanup = cleanup
        return tiny_model(), NumericTokenizer()

    with patch("benchmarks.efficency_vram.pred.load_model_and_tokenizer", side_effect=load) as loader, \
         patch("benchmarks.efficency_vram.pred.EfficiencyMeter", wraps=EfficiencyMeter) as timer:
        paths = run(args)
        assert loader.call_count == 1 and timer.call_count == 3 and cleanup.call_count == 1
        with pytest.raises(FileExistsError):
            run(args)
        assert loader.call_count == 1
        args.overwrite = True
        assert run(args) == paths
    rows = [json.loads(line) for line in paths["metrics"].read_text().splitlines()]
    assert [row["indices"] for row in rows] == [[0, 1], [2, 3], [4]]
    assert [row["batch_size"] for row in rows] == [2, 2, 1]
    summary = json.loads(paths["summary"].read_text())
    assert summary["num_samples"] == 5 and summary["num_batches"] == 3
    assert [group["batch_size"] for group in summary["groups"]] == [1, 2]
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["status"] == "complete" and metadata["effective_method_config"]["sink_size"] == 16
    assert len(paths["predictions"].read_text().splitlines()) == 5
    assert resolve_data_file(args) == data_file
    assert summarize([rows[-1]])["groups"][0]["batch_size"] == 1


def test_prediction_only_entrypoint_saves_predictions(tmp_path):
    data_file = tmp_path / "validation.jsonl"
    data_file.write_text("\n".join(json.dumps({"index": index, "input": "1 2 3"}) for index in range(5)))
    args = parse_args(["--model", "qwen3-4b", "--measurement", "none", "--device", "cpu",
                       "--data-dir", str(data_file), "--output-root", str(tmp_path / "output"),
                       "--input-length", "4", "--fixed-output-length", "2", "--warmup", "0",
                       "--num-samples", "-1", "--num-batches", "2"])

    def load(options):
        options.model_path = "local-tiny-qwen3"
        return tiny_model(), NumericTokenizer()

    with patch("benchmarks.efficency_vram.pred.load_model_and_tokenizer", side_effect=load), \
         patch("benchmarks.efficency_vram.pred.EfficiencyMeter") as timer, \
         patch("benchmarks.efficency_vram.pred.MemoryMonitor") as memory:
        paths = run(args)
    timer.assert_not_called()
    memory.assert_not_called()
    rows = [json.loads(line) for line in paths["predictions"].read_text().splitlines()]
    assert [row["index"] for row in rows] == [0, 1]
    assert all(len(row["output_token_ids"]) == 2 for row in rows)
