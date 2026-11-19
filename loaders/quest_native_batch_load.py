"""Load the independent original-style batched Quest path for Llama inference."""

from argparse import Namespace
from pathlib import Path
import sys

import torch
import yaml

from benchmarks.common import parse_set


def load_model_and_tokenizer(args):
    from transformers import AutoConfig, AutoTokenizer

    root = Path(__file__).resolve().parents[1]
    values = yaml.safe_load((root / 'method_configs/quest_native_batch.yaml').read_text())
    if getattr(args, 'budget', None) is not None:
        values['budget'] = args.budget
    overrides = parse_set(getattr(args, 'set', []), 'quest_native_batch')
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError(f'Unsupported quest_native_batch settings: {sorted(unknown)}')
    values.update(overrides)
    method_args = Namespace(**values)
    if method_args.dtype != 'float16':
        raise ValueError('Native batched Quest currently requires float16')
    config = AutoConfig.from_pretrained(args.model_path)
    if config.model_type != 'llama':
        raise ValueError('Native batched Quest currently supports Llama models')
    quest_root = str(root / 'methods/quest')
    if quest_root not in sys.path:
        sys.path.insert(0, quest_root)
    from quest.native_batch import NativeBatchLlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    model = NativeBatchLlamaForCausalLM.from_pretrained(
        args.model_path, config=config, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation='eager',
        device_map=torch.device(args.device),
    ).eval()
    model.quest_init(batch_size=int(args.batch_size),
        max_seq_len=int(args.max_seq_length) + int(args.fixed_output_length),
        token_budget=int(method_args.budget), page_size=int(method_args.chunk_size),
        skip_layers=int(method_args.skip_layers),
        legacy_head_selection=bool(method_args.legacy_head_selection))
    args._method_args = method_args
    args._quest_model_type = 'native_batch'
    args._cleanup = model.quest_clear
    print(f'Quest backend: original-style native batch CUDA; batch={args.batch_size}, '
          f'budget={method_args.budget}, page_size={method_args.chunk_size}', flush=True)
    return model, tokenizer
