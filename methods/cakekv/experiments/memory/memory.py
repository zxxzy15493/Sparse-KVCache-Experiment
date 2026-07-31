import os
import sys
import json
import random
import argparse
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from cake.cake_cache import CakeprefillKVCache
from cake.utils import CompressConfig


def parse_args():
	parser = argparse.ArgumentParser(description="CakeKV memory benchmark")
	parser.add_argument('--model', type=str, default="llama3.1-8b-128k", choices=["llama3.1-8b-128k", "qwen2.5-7b-instruct"])
	parser.add_argument(
		'--models',
		nargs='+',
		default=None,
		help='models',
	)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--cache_size', type=int, default=1024)
	parser.add_argument('--window_size', type=int, default=32)
	parser.add_argument('--gamma', type=float, default=200.0)
	parser.add_argument('--tau1', type=float, default=1.0)
	parser.add_argument('--tau2', type=float, default=1.0)
	parser.add_argument('--input_file', type=str, required=True)
	parser.add_argument('--save_dir', type=str, default="./results")
	parser.add_argument('--warmup', type=int, default=3)
	parser.add_argument('--input_max_token', type=int, default=4096)
	parser.add_argument(
		'--input_max_tokens',
		nargs='+',
		type=int,
		default=None,
	)
	parser.add_argument('--max_new_tokens', type=int, default=32)
	parser.add_argument('--seed', type=int, default=42)
	return parser.parse_args()


def seed_everything(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.cuda.manual_seed_all(seed)


def load_dataset(input_file):
	with open(input_file, 'r', encoding='utf-8') as f:
		return f.read()


def load_model_and_tokenizer(model_path, device, compress_config):
	config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
	tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

	model_name_lower = model_path.lower()
	if "llama" in model_name_lower:
		from cake.monkeypatch import replace_flashllama_attn_with_cakeattn
		replace_flashllama_attn_with_cakeattn()
	elif "qwen2" in model_name_lower or "qwen" in model_name_lower:
		from cake.monkeypatch import replace_flashqwen2_attn_with_cakeattn
		replace_flashqwen2_attn_with_cakeattn()

	model = AutoModelForCausalLM.from_pretrained(
		model_path,
		torch_dtype=torch.bfloat16,
		attn_implementation="flash_attention_2",
		trust_remote_code=True,
	).to(device)

	model.config.use_cache = True
	if hasattr(model, "generation_config"):
		model.generation_config.use_cache = True

	if tokenizer.eos_token is not None:
		tokenizer.pad_token = tokenizer.eos_token
	elif tokenizer.pad_token_id is None:
		eos_id = model.config.eos_token_id
		if isinstance(eos_id, (list, tuple)) and len(eos_id) > 0:
			tokenizer.pad_token_id = int(eos_id[0])
		elif eos_id is not None:
			tokenizer.pad_token_id = int(eos_id)
	tokenizer.padding_side = "left"

	layers = config.num_hidden_layers
	for i in range(layers):
		attn = model.model.layers[i].self_attn
		attn.config.key_size = [compress_config.cache_size - compress_config.window_size] * layers
		attn.config.window_size = [compress_config.window_size] * layers
		attn.config.prefill = [True] * layers
		attn.config.decoding_evict = [None] * layers
		attn.config.tau1 = compress_config.hyper[0]
		attn.config.tau2 = compress_config.hyper[1]
		attn.config.gamma = compress_config.hyper[2]
		attn.config.prefill_cake_evict = [CakeprefillKVCache(
			cache_size=compress_config.cache_size,
			window_size=compress_config.window_size,
			k_seq_dim=2,
			v_seq_dim=2,
			num_heads=attn.num_heads,
			num_layers=layers,
			use_cascading=compress_config.cascading
		)] * layers

	model = model.eval()
	return model, tokenizer


def measure_memory(model, tokenizer, text, device, input_max_token, max_new_tokens):
	inputs = tokenizer(
		[text],
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=input_max_token,
	)
	input_ids = inputs.input_ids.to(device)
	attention_mask = inputs.attention_mask.to(device)

	print("=" * 80)
	print(f"Input tokens: {input_ids.shape[1]}")
	print("=" * 80)

	warmup_count = 2
	measure_count = 2

	gen_eos_id = tokenizer.eos_token_id
	if gen_eos_id is None:
		gen_eos_id = getattr(model.config, "eos_token_id", None)
	if isinstance(gen_eos_id, (list, tuple)) and len(gen_eos_id) > 0:
		gen_eos_id = int(gen_eos_id[0])

	gen_pad_id = tokenizer.pad_token_id
	if gen_pad_id is None and gen_eos_id is not None:
		gen_pad_id = int(gen_eos_id)

	# Warmup runs
	for _ in range(warmup_count):
		torch.cuda.empty_cache()
		with torch.inference_mode():
			gen_kwargs = {
				"input_ids": input_ids,
				"attention_mask": attention_mask,
				"max_new_tokens": max_new_tokens,
				"num_beams": 1,
				"do_sample": False,
				"temperature": 1.0,
				"use_cache": True,
			}
			if gen_pad_id is not None:
				gen_kwargs["pad_token_id"] = int(gen_pad_id)
			if gen_eos_id is not None:
				gen_kwargs["eos_token_id"] = int(gen_eos_id)
			model.generate(**gen_kwargs)

	torch.cuda.memory.reset_peak_memory_stats()
	torch.cuda.memory._record_memory_history()

	# Measured runs
	for i in range(measure_count):
		torch.cuda.empty_cache()
		with torch.inference_mode():
			gen_kwargs = {
				"input_ids": input_ids,
				"attention_mask": attention_mask,
				"max_new_tokens": max_new_tokens,
				"num_beams": 1,
				"do_sample": False,
				"temperature": 1.0,
				"use_cache": True,
			}
			if gen_pad_id is not None:
				gen_kwargs["pad_token_id"] = int(gen_pad_id)
			if gen_eos_id is not None:
				gen_kwargs["eos_token_id"] = int(gen_eos_id)
			out = model.generate(**gen_kwargs)
		torch.cuda.synchronize()

		generated_tokens = int(out.shape[1] - input_ids.shape[1])
		print(f"[Measured Round {i}] Generated {generated_tokens} tokens")

	peak_alloc = torch.cuda.memory.max_memory_allocated() / 1024**2

	summary = {
		"peak_memory_mib": peak_alloc,
		"effective_rounds": measure_count,
	}

	return summary


def normalize_model_names(args):
	if args.models:
		return args.models
	return [args.model]


def normalize_input_lengths(args):
	if args.input_max_tokens:
		return args.input_max_tokens
	return [args.input_max_token]


def resolve_model_path(model_name_or_path, model2path):
	if os.path.isdir(model_name_or_path) or "/" in model_name_or_path:
		return model_name_or_path
	return model2path.get(model_name_or_path, model_name_or_path)


def main():
	args = parse_args()
	seed_everything(args.seed)

	compress_config = CompressConfig(True, False)
	compress_config.cache_size = args.cache_size
	compress_config.window_size = args.window_size
	compress_config.hyper = [args.tau1, args.tau2, args.gamma]

	device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
	model2path = {
		"llama3.1-8b-128k": "meta-llama/Llama-3.1-8B-Instruct",
		"qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct-1M",
	}

	text = load_dataset(args.input_file)

	for model_name in normalize_model_names(args):
		model_path = resolve_model_path(model_name, model2path)
		model, tokenizer = load_model_and_tokenizer(model_path, device, compress_config)
		for input_max_token in normalize_input_lengths(args):
			method_dir = "cakekv"
			model_slug = os.path.basename(str(model_name).rstrip("/")).replace(" ", "_")
			output_dir = os.path.join(args.save_dir, method_dir, model_slug, f"input_{input_max_token}")
			os.makedirs(output_dir, exist_ok=True)

			summary = measure_memory(model, tokenizer, text, device, input_max_token, args.max_new_tokens)

			out_file = os.path.join(
				output_dir,
				f"memory_{args.cache_size}_{input_max_token}_{args.max_new_tokens}.json",
			)

			meta = {
				"model": model_name,
				"cache_size": args.cache_size,
				"window_size": args.window_size,
				"gamma": args.gamma,
				"tau1": args.tau1,
				"tau2": args.tau2,
				"input_file": args.input_file,
				"input_max_token": input_max_token,
				"max_new_tokens": args.max_new_tokens,
				"warmup": args.warmup,
			}

			with open(out_file, "w", encoding="utf-8") as f:
				json.dump({"meta": meta, "summary": summary}, f, ensure_ascii=False, indent=2)

			print("=" * 80)
			print(f"Saved: {out_file}")
			print(f"Peak Memory: {summary['peak_memory_mib']:.2f} MiB")
			print("=" * 80)


if __name__ == "__main__":
	main()
