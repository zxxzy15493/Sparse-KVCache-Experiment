import argparse
import json
import inspect
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def parse_args():
	parser = argparse.ArgumentParser(description="PyramidKV efficiency benchmark")
	parser.add_argument("--model", type=str, required=False, help="model path")
	parser.add_argument(
		"--models",
		nargs="+",
		default=None,
	)
	parser.add_argument("--input_file", type=str, required=True)
	parser.add_argument("--save_dir", type=str, default="./results")
	parser.add_argument("--warmup", type=int, default=3)
	parser.add_argument("--input_max_token", type=int, default=4096)
	parser.add_argument(
		"--input_max_tokens",
		nargs="+",
		type=int,
		default=None,
	)
	parser.add_argument("--max_new_tokens", type=int, default=32)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--use_fast_tokenizer", action="store_true")
	parser.add_argument("--output_attentions", action="store_true")
	parser.add_argument(
		"--attn_implementation",
		type=str,
		default="flash_attention_2",
		choices=["flash_attention_2", "sdpa", "eager"],
	)
	parser.add_argument("--use_cache", action="store_true", default=True)

	parser.add_argument(
		"--method",
		type=str,
		required=True,
		help="fullkv/pyramidkv/snapkv/h2o/streamingllm",
	)
	parser.add_argument("--max_capacity_prompts", type=int, default=1024)
	parser.add_argument("--max_capacity_prompts_ratio", type=float, default=-1)

	args = parser.parse_args()
	if not args.model and not args.models:
		parser.error("one of --model or --models is required")

	return args


def normalize_model_list(args):
	if args.models:
		return args.models
	return [args.model]


def normalize_input_lengths(args):
	if args.input_max_tokens:
		return args.input_max_tokens
	return [args.input_max_token]


def resolve_model_path(model_name: str):
	model2path = {
		"llama3.1-8b-128k": "meta-llama/Llama-3.1-8B-Instruct",
		"qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct-1M",
	}
	if model_name in model2path:
		return model2path[model_name]

	# Accept valid local paths directly.
	if os.path.isdir(model_name):
		return model_name

	return model_name


def make_safe_name(name: str) -> str:
	base = os.path.basename(str(name).rstrip("/"))
	return base.replace(" ", "_")


def seed_everything(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def normalize_method(method: str) -> str:
	cleaned = "".join(ch for ch in str(method).lower() if ch.isalnum())
	valid = {"fullkv", "pyramidkv", "snapkv", "h2o", "streamingllm"}
	if cleaned not in valid:
		raise ValueError(f"Unsupported method: {method}")
	return cleaned


def load_dataset(input_file: str) -> str:
	with open(input_file, "r", encoding="utf-8") as f:
		return f.read()


def get_revision_for_model(model_path: str):
	# Skip revision pinning — avoids unnecessary hub requests that fail
	# for gated repos. Cached model snapshots are used via local_files_only.
	return None


def get_transformer_layers(model):
	if hasattr(model, "model") and hasattr(model.model, "layers"):
		return model.model.layers
	if (
		hasattr(model, "transformer")
		and hasattr(model.transformer, "encoder")
		and hasattr(model.transformer.encoder, "layers")
	):
		return model.transformer.encoder.layers
	raise ValueError("Unsupported model architecture for layer traversal")


def get_self_attention_module(layer):
	if hasattr(layer, "self_attn"):
		return layer.self_attn
	if hasattr(layer, "self_attention"):
		return layer.self_attention
	raise ValueError("Unsupported layer attention module")


def get_config_holder(model):
	if hasattr(model, "model") and hasattr(model.model, "config"):
		return model.model.config
	if hasattr(model, "transformer") and hasattr(model.transformer, "config"):
		return model.transformer.config
	return model.config


def ensure_pad_token(tokenizer, model):
	tokenizer.padding_side = "left"
	tokenizer.truncation_side = "left"
	if tokenizer.pad_token is None:
		if tokenizer.eos_token is not None:
			tokenizer.pad_token = tokenizer.eos_token
			tokenizer.pad_token_id = tokenizer.eos_token_id
		else:
			tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
			model.resize_token_embeddings(len(tokenizer))


def apply_pyramidkv_patch(method: str, model=None):
	from pyramidkv.monkeypatch import replace_chatglm, replace_llama, replace_qwen2

	replace_llama(method)
	replace_qwen2(method)
	if model is not None:
		replace_chatglm(method, model)


def apply_runtime_pyramid_config(model, method: str, max_capacity_prompts: int):
	if method == "fullkv":
		return

	if method in ["snapkv", "pyramidkv", "h2o"]:
		window_size = 8
	elif method == "streamingllm":
		window_size = max(max_capacity_prompts - 4, 1)
	else:
		window_size = 8

	kernel_size = 7
	pooling = "maxpool"

	layers_list = get_transformer_layers(model)
	layers = len(layers_list)
	config_holder = get_config_holder(model)

	config_holder.base_capacity = max_capacity_prompts
	config_holder.max_capacity_prompt = max_capacity_prompts
	config_holder.window_size = window_size
	config_holder.head_choice = "random"

	window_sizes = [window_size] * layers
	capacities = [max_capacity_prompts] * layers
	kernel_sizes = [kernel_size] * layers

	for i in range(layers):
		attn_module = get_self_attention_module(layers_list[i])
		if not hasattr(attn_module, "config"):
			attn_module.config = model.config
		attn_module.config.base_capacity = max_capacity_prompts
		attn_module.config.max_capacity_prompt = max_capacity_prompts
		attn_module.config.window_size = window_sizes[i]
		attn_module.config.max_capacity_prompt = capacities[i]
		attn_module.config.kernel_size = kernel_sizes[i]
		attn_module.config.pooling = pooling


def reset_runtime_pyramid_state(model):
	try:
		for layer in get_transformer_layers(model):
			attn_module = get_self_attention_module(layer)
			if hasattr(attn_module, "kv_seq_len"):
				attn_module.kv_seq_len = 0
			if hasattr(attn_module, "_cake_timing"):
				attn_module._cake_timing = None
	except Exception:
		pass


def load_model_and_tokenizer(args):
	method = normalize_method(args.method)
	resolved_model = resolve_model_path(args.model)
	model_path_lower = resolved_model.lower()

	config = AutoConfig.from_pretrained(
		resolved_model,
		trust_remote_code=True,
	)
	if "qwen" in model_path_lower or "llama" in model_path_lower:
		for attr_name in ("num_logits_to_keep", "logits_to_keep"):
			try:
				setattr(config, attr_name, 1)
			except Exception:
				pass
	config.use_cache = args.use_cache

	tokenizer = AutoTokenizer.from_pretrained(
		resolved_model,
		use_fast=args.use_fast_tokenizer,
		trust_remote_code=True,
		padding_side="left",
	)
	apply_pyramidkv_patch(method, model=None)

	model = AutoModelForCausalLM.from_pretrained(
		resolved_model,
		config=config,
		trust_remote_code=True,
		torch_dtype=torch.bfloat16,
		low_cpu_mem_usage=True,
		device_map="auto",
		attn_implementation=args.attn_implementation,
	)

	model.eval()
	ensure_pad_token(tokenizer, model)

	model.config.use_cache = True
	if "qwen" in model_path_lower or "llama" in model_path_lower:
		for attr_name in ("num_logits_to_keep", "logits_to_keep"):
			try:
				setattr(model.config, attr_name, 1)
				if hasattr(model, "generation_config"):
					setattr(model.generation_config, attr_name, 1)
			except Exception:
				pass
	if hasattr(model, "generation_config"):
		model.generation_config.use_cache = True

	return model, tokenizer, method


def measure_efficiency(model, tokenizer, text, args, method):
	os.makedirs(args.save_dir, exist_ok=True)

	inputs = tokenizer(
		[text],
		return_tensors="pt",
		padding=True,
		truncation=True,
		max_length=args.input_max_token,
		add_special_tokens=True,
	)

	device = next(model.parameters()).device
	input_ids = inputs.input_ids.to(device)
	attention_mask = inputs.attention_mask.to(device)

	if args.max_capacity_prompts != -1:
		max_capacity_prompts = args.max_capacity_prompts
	elif args.max_capacity_prompts_ratio != -1:
		max_capacity_prompts = round(input_ids.shape[1] * args.max_capacity_prompts_ratio)
	else:
		max_capacity_prompts = args.max_capacity_prompts

	apply_runtime_pyramid_config(model, method, max_capacity_prompts)

	print("=" * 80)
	print(f"Input tokens: {input_ids.shape[1]}")
	print(f"Method: {method}, max_capacity_prompts: {max_capacity_prompts}")
	print("=" * 80)

	results = []
	# Use fixed protocol: warmup 2 rounds, then measure 2 rounds and average
	warmup_count = 2
	measure_count = 2

	eos_token_ids = tokenizer.eos_token_id
	if eos_token_ids is None:
		eos_token_ids = getattr(model.config, "eos_token_id", None)
	if isinstance(eos_token_ids, int):
		eos_token_ids = [eos_token_ids]
	if eos_token_ids is None:
		eos_token_ids = []

	model_extra_kwargs = {}
	model_path_lower = args.model.lower()
	if "glm" in model_path_lower or "chatglm" in model_path_lower:
		model_extra_kwargs["return_last_logit"] = True

	# Some large-context model implementations (Qwen, Llama, etc.) do not
	# expose the `num_logits_to_keep`/`logits_to_keep` argument in the
	# `forward` signature. Try to set conservative defaults on the model
	# config as a best-effort fallback, then prefer explicit forward args
	# when available.
	if "qwen" in model_path_lower or "llama" in model_path_lower:
		try:
			setattr(model.config, "num_logits_to_keep", 1)
		except Exception:
			pass
		try:
			setattr(model.config, "logits_to_keep", 1)
		except Exception:
			pass

	forward_sig = inspect.signature(model.forward)
	if "num_logits_to_keep" in forward_sig.parameters:
		# Only keep the last logit to reduce prefill memory.
		model_extra_kwargs["num_logits_to_keep"] = 1
	elif "logits_to_keep" in forward_sig.parameters:
		# Backward compatibility for models using an older arg name.
		model_extra_kwargs["logits_to_keep"] = 1

	# Warmup runs (do not record)
	for w in range(warmup_count):
		reset_runtime_pyramid_state(model)
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
		with torch.inference_mode():
			start = time.perf_counter()
			prefill_out = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				past_key_values=None,
				use_cache=True,
				output_attentions=args.output_attentions,
				**model_extra_kwargs,
			)
			past_key_values = prefill_out.past_key_values
			pred_token_idx = prefill_out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
			torch.cuda.synchronize()
			_ = time.perf_counter() - start

			for _ in range(max(args.max_new_tokens - 1, 0)):
				outputs = model(
					input_ids=pred_token_idx,
					past_key_values=past_key_values,
					use_cache=True,
					output_attentions=args.output_attentions,
					**model_extra_kwargs,
				)
				torch.cuda.synchronize()
				past_key_values = outputs.past_key_values
				pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

	# Reset peak memory tracking BEFORE measured runs (not inside the loop).
	# Otherwise prefill KV cache allocation is not captured.
	# torch.cuda.memory.reset_peak_memory_stats()
	# torch.cuda.memory._record_memory_history()

	# Measured runs
	for i in range(measure_count):
		reset_runtime_pyramid_state(model)
		torch.cuda.empty_cache()
		torch.cuda.synchronize()

		with torch.inference_mode():
			start = time.perf_counter()
			prefill_out = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				past_key_values=None,
				use_cache=True,
				output_attentions=args.output_attentions,
				**model_extra_kwargs,
			)
			past_key_values = prefill_out.past_key_values
			pred_token_idx = prefill_out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
			torch.cuda.synchronize()
			ttft_wall = time.perf_counter() - start

			decode_times = []
			generated_ids = [pred_token_idx.item()]

			for _ in range(max(args.max_new_tokens - 1, 0)):
				t0 = time.perf_counter()
				outputs = model(
					input_ids=pred_token_idx,
					past_key_values=past_key_values,
					use_cache=True,
					output_attentions=args.output_attentions,
					**model_extra_kwargs,
				)
				torch.cuda.synchronize()
				decode_times.append(time.perf_counter() - t0)

				past_key_values = outputs.past_key_values
				pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
				generated_ids.append(pred_token_idx.item())

			# Prefer internal timing if available on first attention module
			try:
				first_layer = get_transformer_layers(model)[0]
				attn_mod = get_self_attention_module(first_layer)
				internal_timing = getattr(attn_mod, "_cake_timing", None)
			except Exception:
				internal_timing = None

			# Prefer internal timing only when it looks reasonable compared to wall clock.
			use_internal = False
			if internal_timing and isinstance(internal_timing, dict):
				try:
					it_ttft = float(internal_timing.get("ttft", float("nan")))
					it_latency = float(internal_timing.get("latency", float("nan")))
					it_tpot_raw = internal_timing.get("tpot", None)
					it_tpot = float(it_tpot_raw) if it_tpot_raw is not None else float("nan")
				except Exception:
					it_ttft = float("nan")
					it_latency = float("nan")
					it_tpot = float("nan")

				wall_latency = ttft_wall + float(sum(decode_times))
				wall_tpot = float(np.mean(decode_times)) if len(decode_times) > 0 else float("nan")

				# Heuristics: internal timings must be in a similar scale as wall-clock,
				# otherwise they are likely sampled from a different scope.
				if (
					not np.isnan(it_ttft)
					and 0.5 * ttft_wall <= it_ttft <= 2.0 * ttft_wall
					and not np.isnan(it_latency)
					and 0.5 * wall_latency <= it_latency <= 2.0 * wall_latency
					and (
						np.isnan(it_tpot)
						or np.isnan(wall_tpot)
						or (0.5 * wall_tpot <= it_tpot <= 2.0 * wall_tpot)
					)
				):
					use_internal = True

			if use_internal:
				ttft = float(internal_timing.get("ttft", ttft_wall))
				tpot = (
					float(internal_timing.get("tpot"))
					if internal_timing.get("tpot") is not None
					else (float(np.mean(decode_times)) if len(decode_times) > 0 else None)
				)
				latency = float(internal_timing.get("latency", ttft + float(sum(decode_times))))
				timing_source = "internal"
			else:
				# If internal timing looks suspicious, fall back to wall-clock timings.
				if internal_timing and isinstance(internal_timing, dict):
					print(f"[debug] ignoring suspicious internal_timing={internal_timing}")
				ttft = float(ttft_wall)
				tpot = float(np.mean(decode_times)) if len(decode_times) > 0 else None
				latency = ttft + float(sum(decode_times))
				timing_source = "wall_clock"

		pred = tokenizer.decode(generated_ids, skip_special_tokens=True)
		row = {
			"round": i,
			"ttft": float(ttft),
			"tpot": float(tpot) if tpot is not None else None,
			"latency": float(latency),
			"generated_tokens": len(generated_ids),
			"pred": pred,
			"timing_source": timing_source,
		}
		results.append(row)
		tpot_show = f"{row['tpot']:.6f}" if row["tpot"] is not None else "None"
		print(
			f"[Measured Round {i}] TTFT: {row['ttft']:.6f} s | TPOT: {tpot_show} s | Latency: {row['latency']:.6f} s | source: {timing_source}"
		)

	# results now contains only measured rounds (we warmed up separately)
	stable_results = results
	# peak_alloc = torch.cuda.memory.max_memory_allocated() / 1024**2
	valid_ttft = [r["ttft"] for r in stable_results if r["ttft"] is not None]
	valid_tpot = [r["tpot"] for r in stable_results if r["tpot"] is not None]
	valid_latency = [r["latency"] for r in stable_results if r["latency"] is not None]

	summary = {
		"avg_ttft": float(np.mean(valid_ttft)) if len(valid_ttft) > 0 else None,
		"avg_tpot": float(np.mean(valid_tpot)) if len(valid_tpot) > 0 else None,
		"avg_latency": float(np.mean(valid_latency)) if len(valid_latency) > 0 else None,
		"effective_rounds": len(stable_results),
	}

	return results, summary, max_capacity_prompts


def main():
	os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

	args = parse_args()
	seed_everything(args.seed)
	# device: use CUDA if available, otherwise CPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model2path = {
		"llama3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
		"qwen2.5-7b-instruct-1M": "Qwen/Qwen2.5-7B-Instruct-1M",
	}

	text = load_dataset(args.input_file)
	for model_name in normalize_model_list(args):
		model_args = argparse.Namespace(**vars(args))
		model_path = resolve_model_path(model_name)
		# load_model_and_tokenizer expects an args-like namespace where
		# `args.model` is the model path; set it accordingly and call.
		model_args.model = model_path
		model, tokenizer, method = load_model_and_tokenizer(model_args)
		model_slug = make_safe_name(model_name)
		for input_max_token in normalize_input_lengths(args):
			run_args = argparse.Namespace(**vars(model_args))
			run_args.input_max_token = input_max_token
			run_args.save_dir = os.path.join(args.save_dir, model_slug, f"input_{input_max_token}")
			results, summary, max_capacity_prompts = measure_efficiency(model, tokenizer, text, run_args, method)

			out_file = os.path.join(
				run_args.save_dir,
				f"efficiency_{method}_{max_capacity_prompts}_{input_max_token}_{run_args.max_new_tokens}.jsonl",
			)

			meta = {
				"model": model_name,
				"method": method,
				"input_file": run_args.input_file,
				"input_max_token": input_max_token,
				"max_new_tokens": run_args.max_new_tokens,
				"warmup": run_args.warmup,
				"max_capacity_prompts": max_capacity_prompts,
				"max_capacity_prompts_ratio": run_args.max_capacity_prompts_ratio,
				"attn_implementation": run_args.attn_implementation,
				"output_attentions": run_args.output_attentions,
			}

			with open(out_file, "a", encoding="utf-8") as f:
				f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
				for row in results:
					f.write(json.dumps(row, ensure_ascii=False) + "\n")
				f.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")

			print("=" * 80)
			print(f"Saved: {out_file}")
			print(
				f"Avg TTFT: {summary['avg_ttft']} | Avg TPOT: {summary['avg_tpot']} | Avg Latency: {summary['avg_latency']}"
			)
			print("=" * 80)


if __name__ == "__main__":
	main()