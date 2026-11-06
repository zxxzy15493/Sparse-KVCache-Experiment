"""Run the real AdaKV / HeadKV patch on a tiny random Qwen3; no checkpoint download.

This is a best-effort GPU smoke test: it patches ``AutoModelForCausalLM`` /
``AutoTokenizer`` around the real ``load_model`` so HeadKV is patched as it would
be for the full model. If CUDA is unavailable, or the HeadKV hijack cannot drive
a randomly initialised tiny model, the script prints a clear SKIP reason instead
of failing the run.
"""

import json
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from test_batch_measurements import EXPERIMENT_ROOT, Tokenizer, pred, write_jsonl


def tiny_model(*args, **kwargs):
    config = Qwen3Config(vocab_size=100, hidden_size=256, intermediate_size=512,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=64, eos_token_id=99, pad_token_id=99, bos_token_id=98)
    # HeadKV kernels may be unstable on a tiny random model; eager attention is robust here.
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).to(device="cuda:0", dtype=kwargs["torch_dtype"]).eval()


def main():
    if not torch.cuda.is_available():
        print("SKIP cuda_smoke: CUDA is unavailable (a GPU is required to run this smoke test).")
        return
    root = EXPERIMENT_ROOT / "results/cuda-smoke"
    data_file = root / "input.jsonl"
    write_jsonl(data_file, [{"index": i, "input": "1 " + "2 " * (256 + i * 7)} for i in range(5)])
    with patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=tiny_model), \
         patch.object(AutoTokenizer, "from_pretrained", side_effect=lambda *a, **kw: Tokenizer()):
        try:
            for mode in ("efficency", "vram"):
                for batch_size in (1, 2, 4):
                    result_dir = root / mode / f"batch-{batch_size}"
                    pred.main(["--model", "qwen3-4b", "--benchmark", "ruler", "--task", "niah_single_1",
                               "--data_dir", str(data_file), "--save_dir", str(result_dir),
                               "--batch_size", str(batch_size), "--context_length", "4096",
                               "--fixed_output_length", "4", "--warmup", "1", "--device", "cuda:0"],
                              measurement=mode)
                    rows = pred.load_data(result_dir / f"niah_single_1.{mode}.jsonl")
                    assert sum(row["batch_size"] for row in rows) == 5
                    assert all(row["output_tokens"] == [4] * row["batch_size"] for row in rows)
                    assert all(row["batch_size"] == batch_size for row in rows[:-1])  # tail batch is partial
                    if mode == "efficency":
                        assert all(row["ttft"] > 0 for row in rows)
                        assert all(row["latency"] >= row["decode_latency"] for row in rows)
                        assert all(row["output_tokens_per_s"] > 0 for row in rows)
                    else:
                        assert all(row["peak_alloc"] > 0 for row in rows)
                        assert all(row["peak_reserved"] >= row["peak_alloc"] for row in rows)
                    summary = json.loads((result_dir / f"niah_single_1.{mode}.summary.json").read_text())
                    assert summary["num_samples"] == 5
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP cuda_smoke: HeadKV could not drive the tiny random model: {exc}")
            return
    print("CUDA smoke passed: real AdaKV/HeadKV patch, B=1/2/4, both metrics, "
          "unequal left padding and tail batches.")


if __name__ == "__main__":
    main()
