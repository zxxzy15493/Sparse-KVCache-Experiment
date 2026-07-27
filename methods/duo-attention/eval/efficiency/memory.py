import argparse
import gc
import json
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from duo_attn.patch import enable_duo_attention_eval
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache
from duo_attn.utils import load_attn_pattern, sparsify_attention_heads, to_device


def parse_args():
	parser = argparse.ArgumentParser(description="DuoAttention memory benchmark")
	parser.add_argument("--model", type=str, required=True, help="model name or path")
	parser.add_argument(
		"--models",
		nargs="+",
		default=None,
	)
	parser.add_argument(
		"--model_map",
		type=str,
		default=None,
		help="optional path to model2path.json",
	)
	parser.add_argument("--device", type=str, default="cuda:0")
	parser.add_argument("--input_file", type=str, required=True)
	parser.add_argument("--save_dir", type=str, default="./results")
	parser.add_argument("--warmup", type=int, default=3)
	parser.add_argument("--input_max_token", type=int, default=1024)
	parser.add_argument(
		"--input_max_tokens",
		nargs="+",
		type=int,
		default=None,
	)
	parser.add_argument("--max_new_tokens", type=int, default=8)
	parser.add_argument("--seed", type=int, default=42)

	parser.add_argument("--method", type=str, default="full", choices=["full", "duo_attn"])
	parser.add_argument("--attn_load_dir", type=str, default=None)
	parser.add_argument("--sink_size", type=int, default=64)
	parser.add_argument("--recent_size", type=int, default=256)
	parser.add_argument("--sparsity", type=float, default=0.5)
	parser.add_argument(
		"--sparsities",
		nargs="+",
		type=float,
		default=None,
		help="batch sparsity list",
	)

	return parser.parse_args()


def seed_everything(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def resolve_model_path(model_name_or_path: str, model_map_path: str | None = None) -> str:
	if os.path.isdir(model_name_or_path) or "/" in model_name_or_path:
		return model_name_or_path

	if model_map_path is None:
		cur_dir = os.path.dirname(os.path.abspath(__file__))
		model_map_path = os.path.join(cur_dir, "..", "LongBench", "config", "model2path.json")

	if os.path.exists(model_map_path):
		with open(model_map_path, "r", encoding="utf-8") as f:
			model_map = json.load(f)
		if model_name_or_path in model_map:
			return model_map[model_name_or_path]

	return model_name_or_path


def load_dataset(input_file: str) -> str:
	with open(input_file, "r", encoding="utf-8") as f:
		return f.read()


def normalize_model_list(args):
	if args.models:
		return args.models
	return [args.model]


def normalize_input_lengths(args):
	if args.input_max_tokens:
		return args.input_max_tokens
	return [args.input_max_token]


def normalize_sparsities(args):
	if args.sparsities:
		return args.sparsities
	return [args.sparsity]


def make_safe_name(name: str) -> str:
	base = os.path.basename(str(name).rstrip("/"))
	return base.replace(" ", "_")


def ensure_pad_token(tokenizer, model):
	if tokenizer.eos_token is not None and tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	if tokenizer.pad_token_id is None:
		eos_id = getattr(model.config, "eos_token_id", None)
		if isinstance(eos_id, (list, tuple)):
			eos_id = eos_id[0] if len(eos_id) > 0 else None
		tokenizer.pad_token_id = int(eos_id) if eos_id is not None else 0
	tokenizer.padding_side = "left"


def load_model_and_tokenizer(args):
	model_path = resolve_model_path(args.model, args.model_map)
	tokenizer = AutoTokenizer.from_pretrained(
		model_path,
		trust_remote_code=True,
		use_fast=False,
	)
	model = AutoModelForCausalLM.from_pretrained(
		model_path,
		trust_remote_code=True,
		torch_dtype=torch.bfloat16,
		low_cpu_mem_usage=True,
		attn_implementation="flash_attention_2",
	)
	model.eval()

	if model.config.model_type in ["mistral", "llama"] and hasattr(model, "model"):
		model.model._prepare_decoder_attention_mask = lambda *x, **y: None

	if args.method == "duo_attn":
		if args.attn_load_dir is None:
			raise ValueError("When --method duo_attn, --attn_load_dir must be provided")
		full_attention_heads, sink_size, recent_size = load_attn_pattern(args.attn_load_dir)
		if args.sink_size is not None:
			sink_size = args.sink_size
		if args.recent_size is not None:
			recent_size = args.recent_size

		full_attention_heads, true_sparsity = sparsify_attention_heads(
			full_attention_heads,
			None,
			sparsity=args.sparsity,
		)
		print(f"[Duo] sink_size={sink_size}, recent_size={recent_size}, true_sparsity={true_sparsity}")
		enable_duo_attention_eval(model, full_attention_heads, sink_size, recent_size)
	else:
		sink_size, recent_size, true_sparsity = None, None, None
		enable_tuple_kv_cache(model)

	model = to_device(model, args.device)
	ensure_pad_token(tokenizer, model)

	model.config.use_cache = True
	if hasattr(model, "generation_config"):
		model.generation_config.use_cache = True

	return model, tokenizer, model_path, sink_size, recent_size, true_sparsity


def _to_legacy_kv_if_needed(past_key_values):
	if past_key_values is None:
		return None
	if hasattr(past_key_values, "to_legacy_cache"):
		return past_key_values.to_legacy_cache()
	return past_key_values


def measure_memory(model, tokenizer, text, args, input_max_token: int):
	os.makedirs(args.save_dir, exist_ok=True)

	inputs = tokenizer(
		[text],
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=input_max_token,
	)
	input_ids = inputs.input_ids.to(model.device)
	attention_mask = inputs.attention_mask.to(model.device)

	print("=" * 80)
	print(f"Input tokens: {input_ids.shape[1]}")
	print("=" * 80)

	warmup_count = 2
	measure_count = 2

	torch.cuda.memory._record_memory_history(enabled=None)

	# Warmup runs
	for _ in range(warmup_count):
		torch.cuda.empty_cache()
		torch.cuda.synchronize()

		with torch.inference_mode():
			model_extra_kwargs = {}
			model_type = str(getattr(model.config, "model_type", "")).lower()
			if "chatglm" in model_type or "glm" in model_type:
				model_extra_kwargs["return_last_logit"] = True
			model_name_cfg = str(getattr(model.config, "name_or_path", "") or "").lower()
			if "qwen" in model_type or "qwen" in model_name_cfg:
				model_extra_kwargs["num_logits_to_keep"] = 1

			prefill_out = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				past_key_values=None,
				use_cache=True,
				**model_extra_kwargs,
			)
			past_key_values = _to_legacy_kv_if_needed(prefill_out.past_key_values)
			pred_token_idx = prefill_out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

			eos_token_ids = tokenizer.eos_token_id
			if eos_token_ids is None:
				eos_token_ids = getattr(model.config, "eos_token_id", None)
			if isinstance(eos_token_ids, int):
				eos_token_ids = [eos_token_ids]
			if eos_token_ids is None:
				eos_token_ids = []

			for _ in range(max(args.max_new_tokens - 1, 0)):
				outputs = model(
					input_ids=pred_token_idx,
					past_key_values=past_key_values,
					use_cache=True,
					**model_extra_kwargs,
				)
				torch.cuda.synchronize()

				past_key_values = _to_legacy_kv_if_needed(outputs.past_key_values)
				pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
				if pred_token_idx.item() in eos_token_ids:
					break

	torch.cuda.memory.reset_peak_memory_stats()
	torch.cuda.memory._record_memory_history()

	# Measured runs
	for i in range(measure_count):
		torch.cuda.empty_cache()
		torch.cuda.synchronize()

		with torch.inference_mode():
			model_extra_kwargs = {}
			model_type = str(getattr(model.config, "model_type", "")).lower()
			if "chatglm" in model_type or "glm" in model_type:
				model_extra_kwargs["return_last_logit"] = True
			model_name_cfg = str(getattr(model.config, "name_or_path", "") or "").lower()
			if "qwen" in model_type or "qwen" in model_name_cfg:
				model_extra_kwargs["num_logits_to_keep"] = 1

			prefill_out = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				past_key_values=None,
				use_cache=True,
				**model_extra_kwargs,
			)
			past_key_values = _to_legacy_kv_if_needed(prefill_out.past_key_values)
			pred_token_idx = prefill_out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
			torch.cuda.synchronize()

			generated_ids = [pred_token_idx.item()]

			eos_token_ids = tokenizer.eos_token_id
			if eos_token_ids is None:
				eos_token_ids = getattr(model.config, "eos_token_id", None)
			if isinstance(eos_token_ids, int):
				eos_token_ids = [eos_token_ids]
			if eos_token_ids is None:
				eos_token_ids = []

			for _ in range(max(args.max_new_tokens - 1, 0)):
				outputs = model(
					input_ids=pred_token_idx,
					past_key_values=past_key_values,
					use_cache=True,
					**model_extra_kwargs,
				)
				torch.cuda.synchronize()

				past_key_values = _to_legacy_kv_if_needed(outputs.past_key_values)
				pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
				generated_ids.append(pred_token_idx.item())
				if pred_token_idx.item() in eos_token_ids:
					break

		print(f"[Measured Round {i}] Generated {len(generated_ids)} tokens")

	peak_alloc = torch.cuda.memory.max_memory_allocated() / 1024**2
	torch.cuda.memory._record_memory_history(enabled=None)

	summary = {
		"peak_memory_mib": peak_alloc,
		"effective_rounds": measure_count,
	}

	return summary


def main():
	args = parse_args()
	seed_everything(args.seed)

	text = load_dataset(args.input_file)
	for model_name in normalize_model_list(args):
		model_path = resolve_model_path(model_name, args.model_map)
		model_slug = make_safe_name(model_name)
		for input_max_token in normalize_input_lengths(args):
			for sparsity in normalize_sparsities(args):
				model_args = argparse.Namespace(**vars(args))
				model_args.model = model_name
				model_args.input_max_token = input_max_token
				model_args.sparsity = sparsity

				model, tokenizer, _, sink_size, recent_size, true_sparsity = load_model_and_tokenizer(model_args)
				summary = measure_memory(model, tokenizer, text, model_args, input_max_token)

				output_dir = os.path.join(model_args.save_dir, model_slug, f"input_{input_max_token}")
				os.makedirs(output_dir, exist_ok=True)
				output_file = os.path.join(
					output_dir,
					f"memory_{model_args.method}_{input_max_token}_{model_args.max_new_tokens}_sparsity_{sparsity}.json",
				)

				meta = {
					"model": model_name,
					"model_path": model_path,
					"method": model_args.method,
					"input_file": model_args.input_file,
					"input_max_token": input_max_token,
					"max_new_tokens": model_args.max_new_tokens,
					"warmup": model_args.warmup,
					"sink_size": sink_size,
					"recent_size": recent_size,
					"target_sparsity": model_args.sparsity if model_args.method == "duo_attn" else None,
					"true_sparsity": true_sparsity,
				}

				with open(output_file, "w", encoding="utf-8") as f:
					json.dump({"meta": meta, "summary": summary}, f, ensure_ascii=False, indent=2)

				print("=" * 80)
				print(f"Saved: {output_file}")
				print(f"Peak Memory: {summary['peak_memory_mib']:.2f} MiB")
				print("=" * 80)

				del model
				del tokenizer
				gc.collect()
				torch.cuda.empty_cache()


if __name__ == "__main__":
	main()
