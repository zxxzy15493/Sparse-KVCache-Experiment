import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import csv
import time
from datetime import datetime

import numpy as np
import torch
from loguru import logger
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def patch_llama_flash_attention_no_repeat_kv():
    from transformers.cache_utils import StaticCache
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    from transformers.models.llama import modeling_llama
    from transformers.models.llama.modeling_llama import (
        LlamaFlashAttention2,
        apply_rotary_pos_emb,
    )

    hf_logger = modeling_llama.logger

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor = None,
        position_embeddings=None,
    ):
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            hf_logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            hf_logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    LlamaFlashAttention2.forward = forward


def patch_qwen2_flash_attention_no_repeat_kv():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    from transformers.models.qwen2 import modeling_qwen2
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2FlashAttention2,
        apply_rotary_pos_emb,
    )

    hf_logger = modeling_qwen2.logger

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor = None,
        position_embeddings=None,
    ):
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            hf_logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            kv_seq_len = key_states.shape[-2] + cache_position[0]
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            hf_logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    Qwen2FlashAttention2.forward = forward


def patch_flash_attention_by_model_name(model_name):
    model_name = model_name.lower()
    if "qwen" in model_name:
        patch_qwen2_flash_attention_no_repeat_kv()
        print("Patched Qwen2FlashAttention2.forward: no repeat_kv before flash.", flush=True)
    elif "llama" in model_name:
        patch_llama_flash_attention_no_repeat_kv()
        print("Patched LlamaFlashAttention2.forward: no repeat_kv before flash.", flush=True)


def parse_int_list(values):
    out = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Latency test for full attention")

    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--input-file", type=str, default="./myinput.txt")
    parser.add_argument(
        "--input-lens",
        type=str,
        nargs="+",
        required=True,
        help="Input lengths, e.g. 4096 8192 16384 or 4096,8192,16384",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)

    parser.add_argument("--csv", type=str, default="./latency_results.csv")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--measure-rounds", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=1.0)

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )

    args = parser.parse_args()
    args.input_lens = parse_int_list(args.input_lens)
    return args


def get_dtype(dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model(args):
    dtype = get_dtype(args.dtype)
    if args.attn_implementation == "flash_attention_2":
        patch_flash_attention_by_model_name(args.model)

    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )

    model = model.eval().to(args.device)
    return tokenizer, model, dtype


def build_input(tokenizer, input_file, max_input_len, device):
    with open(input_file, "r", encoding="utf-8") as f:
        input_string = f.read()

    encoded = tokenizer(input_string, truncation=False, return_tensors="pt")
    input_ids = encoded.input_ids

    if input_ids.shape[1] < max_input_len:
        repeat_times = (max_input_len + input_ids.shape[1] - 1) // input_ids.shape[1]
        input_ids = input_ids.repeat(1, repeat_times)

    input_ids = input_ids[:, :max_input_len].to(device)
    return input_ids


def append_csv(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "time",
        "model",
        "input_len",
        "max_new_tokens",
        "budget",
        "avg_ttft_s",
        "avg_decode_per_token_s",
        "avg_total_s",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def timed_generate(model, tokenizer, input_ids, max_new_tokens):
    torch.cuda.synchronize()
    begin = time.perf_counter()

    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )[0]

    torch.cuda.synchronize()
    end = time.perf_counter()
    return end - begin


def main():
    args = parse_args()

    print("=" * 80, flush=True)
    print(f"Model               : {args.model}", flush=True)
    print(f"Input lens          : {args.input_lens}", flush=True)
    print(f"Output len          : {args.max_new_tokens}", flush=True)
    print(f"Warmup/Measure      : {args.warmup_rounds}/{args.measure_rounds}", flush=True)
    print(f"Device              : {args.device}", flush=True)
    print(f"Dtype               : {args.dtype}", flush=True)
    print(f"Attention impl      : {args.attn_implementation}", flush=True)
    print("=" * 80, flush=True)

    tokenizer, model, dtype = load_model(args)

    input_ids_all = build_input(
        tokenizer=tokenizer,
        input_file=args.input_file,
        max_input_len=max(args.input_lens),
        device=args.device,
    )
    print(f"Actual prepared input_ids shape: {tuple(input_ids_all.shape)}", flush=True)

    total_rounds = args.warmup_rounds + args.measure_rounds
    results = {
        seqlen: {
            "ttft": [],
            "total": [],
            "decode_elapsed": [],
            "decode_per_token": [],
        }
        for seqlen in args.input_lens
    }

    for round_idx in range(total_rounds):
        is_warmup = round_idx < args.warmup_rounds
        round_name = "warmup" if is_warmup else "measure"

        print(f"\n===== Round {round_idx + 1}/{total_rounds} ({round_name}) =====", flush=True)

        for seqlen in args.input_lens:
            input_ids = input_ids_all[:, :seqlen]

            if args.sleep > 0:
                time.sleep(args.sleep)
            
            ttft = timed_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=1,
            )
            torch.cuda.empty_cache()
            
            if args.sleep > 0:
                time.sleep(args.sleep)

            total = timed_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
            )

            decode_elapsed = total - ttft
            torch.cuda.empty_cache()

            if args.max_new_tokens > 1:
                decode_per_token = decode_elapsed / (args.max_new_tokens - 1)
            else:
                decode_per_token = 0.0

            print(
                f"input_len={seqlen}, output_len={args.max_new_tokens}, "
                f"ttft={ttft:.6f}s, total={total:.6f}s, "
                f"decode_elapsed={decode_elapsed:.6f}s, "
                f"decode_per_token={decode_per_token:.6f}s",
                flush=True,
            )

            if not is_warmup:
                results[seqlen]["ttft"].append(ttft)
                results[seqlen]["total"].append(total)
                results[seqlen]["decode_elapsed"].append(decode_elapsed)
                results[seqlen]["decode_per_token"].append(decode_per_token)

        torch.cuda.empty_cache()

    print("\n===== Average of measured rounds =====", flush=True)

    for seqlen in args.input_lens:
        ttft_arr = np.array(results[seqlen]["ttft"], dtype=np.float64)
        total_arr = np.array(results[seqlen]["total"], dtype=np.float64)
        decode_per_token_arr = np.array(results[seqlen]["decode_per_token"], dtype=np.float64)

        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "input_len": seqlen,
            "max_new_tokens": args.max_new_tokens,
            "budget": "full",
            "avg_ttft_s": f"{float(np.mean(ttft_arr)):.6f}",
            "avg_decode_per_token_s": f"{float(np.mean(decode_per_token_arr)):.6f}",
            "avg_total_s": f"{float(np.mean(total_arr)):.6f}",
        }

        append_csv(args.csv, row)

        print(
            f"input_len={seqlen}, "
            f"avg_ttft={row['avg_ttft_s']}s, "
            f"avg_total={row['avg_total_s']}s, "
            f"avg_decode_per_token={row['avg_decode_per_token_s']}s",
            flush=True,
        )

    del model
    torch.cuda.empty_cache()
    logger.info("Full attention latency test done.")


if __name__ == "__main__":
    main()
