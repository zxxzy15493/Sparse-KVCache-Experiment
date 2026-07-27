
# ======================== Phase 1: argument parsing (preserve cakekv original) ========================
import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from cake.cake_cache import CakeprefillKVCache
from cake.utils import CompressConfig


# python efficiency.py --models llama3.1-8b-128k qwen2.5-7b-instruct --input_file --cache_size 1024 --input_max_token 131072 --max_new_tokens 32 --save_dir ./results --debug_timing


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default="llama3.1-8b-128k", choices=["llama3.1-8b-128k", "qwen2.5-7b-instruct"])
	parser.add_argument(
		'--models',
		nargs='+',
		default=None,
		help='model list for batch run, e.g. llama3.1-8b-128k qwen2.5-7b-instruct',
	)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--cache_size', type=int, default=1024)
	parser.add_argument('--window_size', type=int, default=32)
	parser.add_argument('--gamma', type=float, default=200.0)
	parser.add_argument('--tau1', type=float, default=1.0)
	parser.add_argument('--tau2', type=float, default=1.0)
	parser.add_argument('--input_file', type=str, default="")
	parser.add_argument('--save_dir', type=str, default="./results")
	parser.add_argument('--warmup', type=int, default=3)
	parser.add_argument('--input_max_token', type=int, default=4096)
	parser.add_argument(
		'--input_max_tokens',
		nargs='+',
		type=int,
		default=None,
		help='4096 8192 16384 32768 65536 131072',
	)
	parser.add_argument('--max_new_tokens', type=int, default=32)
	parser.add_argument('--debug_timing', action='store_true')
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
		text = f.read()
	return text

def load_model_and_tokenizer(model_path, device, compress_config, debug_timing=False):
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
	if tokenizer.pad_token_id is None:
		gen_eos_id = model.generation_config.eos_token_id if hasattr(model, "generation_config") else None
		if isinstance(gen_eos_id, (list, tuple)) and len(gen_eos_id) > 0:
			tokenizer.pad_token_id = int(gen_eos_id[0])
		elif gen_eos_id is not None:
			tokenizer.pad_token_id = int(gen_eos_id)

	layers = config.num_hidden_layers
	for i in range(layers):
		attn = model.model.layers[i].self_attn
		attn.config.key_size = [compress_config.cache_size - compress_config.window_size]*layers
		attn.config.window_size = [compress_config.window_size]*layers
		attn.config.prefill = [True]*layers
		attn.config.decoding_evict = [None]*layers
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
		)]*layers
	model = model.eval()

	if debug_timing:
		first_attn = model.model.layers[0].self_attn
		print("[Debug] model_class:", model.__class__.__name__)
		print("[Debug] first_attn_class:", first_attn.__class__.__name__)
		print("[Debug] tokenizer.pad_token_id:", tokenizer.pad_token_id)
		print("[Debug] tokenizer.eos_token_id:", tokenizer.eos_token_id)
		print("[Debug] model.config.eos_token_id:", getattr(model.config, "eos_token_id", None))
		if hasattr(model, "generation_config"):
			print("[Debug] generation_config.eos_token_id:", getattr(model.generation_config, "eos_token_id", None))
	return model, tokenizer

def get_pred(
	model,
	tokenizer,
	text,
	device,
	save_dir,
	warmup=2,
	input_max_token=1024,
	max_new_tokens=1,
	debug_timing=False,
	model_name=None,
):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
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
	print(f"Input token length: {input_ids.shape[1]}")
	print("=" * 80)

	timing_holders = [
		("model", model),
		("model.model", getattr(model, "model", None)),
		("model.transformer", getattr(model, "transformer", None)),
	]
	timing_holders = [(name, obj) for name, obj in timing_holders if obj is not None]

	gen_eos_id = tokenizer.eos_token_id
	if gen_eos_id is None:
		gen_eos_id = getattr(model.config, "eos_token_id", None)
	if isinstance(gen_eos_id, (list, tuple)) and len(gen_eos_id) > 0:
		gen_eos_id = int(gen_eos_id[0])

	gen_pad_id = tokenizer.pad_token_id
	if gen_pad_id is None and gen_eos_id is not None:
		gen_pad_id = int(gen_eos_id)

	if debug_timing:
		print("[Debug] generate.pad_token_id:", gen_pad_id)
		print("[Debug] generate.eos_token_id:", gen_eos_id)
		print("[Debug] timing_holders:", [name for name, _ in timing_holders])

	results = []
	warmup_count = 2
	measure_count = 2

	for _ in range(warmup_count):
		for _, holder in timing_holders:
			if hasattr(holder, "_cake_timing"):
				holder._cake_timing = None
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
	torch.cuda.memory.reset_peak_memory_stats()
	torch.cuda.memory._record_memory_history()

	for i in range(measure_count):
		for _, holder in timing_holders:
			if hasattr(holder, "_cake_timing"):
				holder._cake_timing = None
		torch.cuda.empty_cache()
		run_start = time.perf_counter()
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
		run_end = time.perf_counter()

		timing = None
		timing_source = None
		for name, holder in timing_holders:
			v = getattr(holder, "_cake_timing", None)
			if v is None:
				continue
			if timing is None:
				timing = v
				timing_source = name
			if isinstance(v, dict) and v.get("ttft") is not None:
				timing = v
				timing_source = name
				break

		if debug_timing:
			for name, holder in timing_holders:
				print(f"[Debug][Round {i}] {name}._cake_timing =", getattr(holder, "_cake_timing", None))
			print(f"[Debug][Round {i}] selected_timing_source =", timing_source)

		ttft = timing["ttft"] if timing and "ttft" in timing else None
		tpot = timing["tpot"] if timing and "tpot" in timing else None
		latency = timing["latency"] if timing and "latency" in timing else None
		generated_tokens = int(out.shape[1] - input_ids.shape[1])
		if latency is None:
			latency = run_end - run_start
		if ttft is None and generated_tokens > 0:
			ttft = latency
		if tpot is None and generated_tokens > 1:
			tpot = max((latency - ttft) / (generated_tokens - 1), 0.0)
		pred = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
		results.append({
			"round": i,
			"ttft": ttft,
			"tpot": tpot,
			"latency": latency,
			"generated_tokens": generated_tokens,
			"pred": pred,
		})
		if ttft is not None and tpot is not None:
			print(f"[Round {i}] TTFT: {ttft:.6f} s | TPOT: {tpot:.6f} s | Latency: {latency:.6f} s")
		else:
			print(f"[Round {i}] No _cake_timing was collected; prediction has been written.")

	stable_results = results
	peak_alloc = torch.cuda.memory.max_memory_allocated() / 1024**2
	valid_ttft = [r["ttft"] for r in stable_results if r["ttft"] is not None]
	valid_tpot = [r["tpot"] for r in stable_results if r["tpot"] is not None]
	valid_latency = [r["latency"] for r in stable_results if r["latency"] is not None]
	avg = {
		"avg_ttft": sum(valid_ttft) / len(valid_ttft) if valid_ttft else None,
		"avg_tpot": sum(valid_tpot) / len(valid_tpot) if valid_tpot else None,
		"avg_latency": sum(valid_latency) / len(valid_latency) if valid_latency else None,
		"peak_alloc": peak_alloc,
		"effective_rounds": len(stable_results),
	}

	file_model_name = os.path.basename(str(model_name).rstrip("/")) if model_name else "model"
	file_model_name = file_model_name.replace(" ", "_")
	out_file = os.path.join(save_dir, f"efficiency_{file_model_name}_{input_max_token}_{max_new_tokens}.jsonl")
	with open(out_file, "a", encoding="utf-8") as f:
		for res in results:
			f.write(json.dumps(res, ensure_ascii=False) + "\n")
		f.write(json.dumps({"summary": avg}, ensure_ascii=False) + "\n")

	print("=" * 80)
	print(f"Results written to: {out_file}")
	print(f"Average TTFT: {avg['avg_ttft']} | Average TPOT: {avg['avg_tpot']} | Average latency: {avg['avg_latency']} | peak_alloc: {avg['peak_alloc']:.2f} MiB")
	print("=" * 80)


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
		"qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct-1M"
	}
	text = load_dataset(args.input_file)

	for model_name in normalize_model_names(args):
		model_path = resolve_model_path(model_name, model2path)
		model, tokenizer = load_model_and_tokenizer(model_path, device, compress_config, debug_timing=args.debug_timing)
		for input_max_token in normalize_input_lengths(args):
			method_dir = getattr(args, 'method', None) or os.environ.get('RESULT_METHOD', 'cakekv')
			method_dir = str(method_dir).replace(' ', '_')
			model_slug = os.path.basename(str(model_name).rstrip("/")).replace(" ", "_")
			output_dir = os.path.join(args.save_dir, method_dir, model_slug, f"input_{input_max_token}")
			get_pred(
				model,
				tokenizer,
				text,
				device,
				output_dir,
				warmup=args.warmup,
				input_max_token=input_max_token,
				max_new_tokens=args.max_new_tokens,
				debug_timing=args.debug_timing,
				model_name=model_name,
			)

if __name__ == "__main__":
	main()
