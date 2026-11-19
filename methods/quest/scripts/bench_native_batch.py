"""Standalone equal-length native Quest benchmark; no efficiency scripts are patched."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from transformers import LlamaConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quest.native_batch import NativeBatchLlamaForCausalLM
from quest.native_batch.ops import load_kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', help='Local Llama checkpoint; omit for a tiny random model')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--prompt-len', type=int, default=1024)
    parser.add_argument('--new-tokens', type=int, default=32)
    parser.add_argument('--budget', type=int, default=1024)
    parser.add_argument('--page-size', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--query-head-selection', action='store_true', help='Opt out of original GQA page-row semantics')
    args = parser.parse_args()
    if min(args.prompt_len, args.new_tokens, args.runs, *args.batch_sizes) < 1 or args.warmup < 0:
        parser.error('Lengths, runs and batch sizes must be positive; warmup must be nonnegative')
    if not torch.cuda.is_available():
        parser.error('CUDA is required')
    torch.manual_seed(17)
    load_kernels()  # Build/import before timing.
    device = torch.device('cuda', torch.cuda.current_device())
    if args.model_path:
        model = NativeBatchLlamaForCausalLM.from_pretrained(args.model_path, dtype=torch.float16,
            low_cpu_mem_usage=True, attn_implementation='eager', local_files_only=True,
            device_map=device)
    else:
        config = LlamaConfig(vocab_size=257, hidden_size=512, intermediate_size=1024,
            num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2)
        model = NativeBatchLlamaForCausalLM(config).half().to(device)
    model = model.eval()
    print(json.dumps({'device': torch.cuda.get_device_name(), 'model': args.model_path or 'tiny-random',
                      'prompt_len': args.prompt_len, 'new_tokens': args.new_tokens,
                      'budget': args.budget, 'legacy_head_selection': not args.query_head_selection}), flush=True)
    for batch in args.batch_sizes:
        model.quest_init(batch, args.prompt_len + args.new_tokens, args.budget,
                         args.page_size, legacy_head_selection=not args.query_head_selection)
        inputs = torch.randint(3, model.config.vocab_size, (batch, args.prompt_len), device='cuda')
        for _ in range(args.warmup):
            model.generate_fixed(inputs, args.new_tokens)
        rows = []
        for run in range(args.runs):
            model.quest_clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            stamps = []
            def stamp():
                torch.cuda.synchronize()
                stamps.append(time.perf_counter())
            start = time.perf_counter()
            output = model.generate_fixed(inputs, args.new_tokens, prefill_hook=stamp, decode_hook=stamp)
            if output.shape != (batch, args.new_tokens):
                raise AssertionError('Unexpected generated token shape')
            torch.cuda.synchronize()
            row = {'batch': batch, 'run': run, 'prefill_ms': round((stamps[0] - start)*1000, 2),
                   'decode_ms': round((stamps[1] - stamps[0])*1000, 2),
                   'decode_step_ms': round((stamps[1] - stamps[0])*1000/max(args.new_tokens - 1, 1), 2),
                   'peak_allocated_gib': round(torch.cuda.max_memory_allocated()/2**30, 2),
                   'plan_builds_total': model.controller.plan_builds}
            rows.append(row)
            print(json.dumps(row), flush=True)
        for field in ('prefill_ms', 'decode_ms', 'decode_step_ms', 'peak_allocated_gib'):
            print(f'batch={batch} mean_{field}={sum(row[field] for row in rows)/len(rows):.2f}', flush=True)
        model.quest_clear()


if __name__ == '__main__':
    main()
