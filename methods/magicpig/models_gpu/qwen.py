"""Token-level Qwen adapter for the AVX-512-free MagicPIG GPU backend."""

import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .attention import configure_qwen_magicpig, register_qwen_magicpig


QWEN25_7B_1M = "Qwen/Qwen2.5-7B-Instruct-1M"


class Qwen2Model:
    """Match models_single's greedy generation API using native HF Qwen KV.

    Qwen2.5-7B-Instruct-1M uses the native Qwen2 architecture. No CPU LSH,
    sparse_attention_cpu, FlashInfer, or separately checked-out MagicPIG is
    needed. The GPU retains the full KV cache and LSH sampling metadata.
    """

    _magicpig_backend = "gpu"

    def __init__(self, model_name=QWEN25_7B_1M, K=0, L=150, batch_size=1,
                 max_length=256, num_sink_tokens=16, num_local_tokens=32,
                 generation_buffer=256, device="cuda:0", dtype=torch.float16,
                 RECALL=False, fixed_output_length=0, prefill_chunk_size=1024):
        if not 0 <= K <= 31 or L < 1:
            raise ValueError("MagicPIG GPU requires 0 <= K <= 31 and L >= 1")
        if min(batch_size, max_length, prefill_chunk_size) < 1:
            raise ValueError("batch_size, max_length and prefill_chunk_size must be positive")
        if min(num_sink_tokens, num_local_tokens, generation_buffer, fixed_output_length) < 0:
            raise ValueError("Sink/window, generation_buffer and fixed_output_length must be nonnegative")
        if RECALL:
            raise ValueError("MagicPIG models_gpu does not implement RECALL analysis")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if config.model_type not in ("qwen2", "qwen3"):
            raise ValueError(f"MagicPIG models_gpu supports dense Qwen2/Qwen3, got {config.model_type!r}")
        self.model_name, self.config = model_name, config
        self.device, self.dtype = torch.device(device), dtype
        self.batch_size, self.max_length = int(batch_size), int(max_length)
        self.generation_buffer, self.chunk_size = int(generation_buffer), int(prefill_chunk_size)
        self.RECALL, self.fixed_output_length = False, int(fixed_output_length)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, trust_remote_code=True, device_map=str(self.device),
            torch_dtype=dtype, attn_implementation=register_qwen_magicpig() if K else "sdpa",
        ).eval()
        self.attention_server = configure_qwen_magicpig(self.model, K, L, num_sink_tokens, num_local_tokens)
        self.attention_server.generation_buffer = self.generation_buffer
        self.attention_server.batch_size = self.batch_size
        self.attention_server.max_length = self.max_length
        self.generation_config = self.model.generation_config
        self.eos_token_ids = self.generation_config.eos_token_id
        if self.eos_token_ids is None:
            self.eos_token_ids = config.eos_token_id
        self.past_key_values = None
        self.sparse, self.sampled_tokens, self.decode_tokens = [], [], []
        self.generate_token = 0
        self.prefill_latency = self.decode_latency = self.TPOT = 0.0
        self.latency = self.Latency = self.lantency = 0.0

    def attention_config(self):
        state = self.attention_server
        return {"K": state.K, "L": state.L,
                "S": state.sink_tokens if state.K else None,
                "W": state.window if state.K else None}

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def clear(self):
        self.past_key_values = None
        self.attention_server.clear_cache()

    @torch.inference_mode()
    def generate(self, input_ids, max_tokens=128, attention_mask=None,
                 eos_token_id=None, pad_token_id=None):
        """Return a flat token list for B=1 or a tensor [B, prefix+output].

        Native HF cache and forward handle the whole batch. Both left and
        right padding are normalized internally; the returned prefix keeps
        the original input layout, as expected by the benchmark loader.
        """
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
        if input_ids.ndim != 2 or not input_ids.shape[1]:
            raise ValueError("MagicPIG input_ids must have shape [batch_size, nonempty sequence]")
        batch, prefix_width = input_ids.shape
        if batch != self.batch_size:
            raise ValueError(f"MagicPIG was initialized for batch_size={self.batch_size}, got {batch}")
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            valid = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("MagicPIG attention_mask must have the same shape as input_ids")
            if not ((attention_mask == 0) | (attention_mask == 1)).all():
                raise ValueError("MagicPIG attention_mask must contain only 0 or 1")
            valid = attention_mask.to(device=self.device, dtype=torch.bool)
        lengths = valid.sum(dim=-1)
        runs = (valid[:, 1:] != valid[:, :-1]).sum(dim=-1) + valid[:, 0] + valid[:, -1]
        if not (runs == 2).all():
            raise ValueError("MagicPIG requires nonempty prompts with contiguous left/right padding")
        max_tokens = int(self.fixed_output_length or max_tokens)
        if max_tokens < 0:
            raise ValueError("MagicPIG max_tokens must be nonnegative")
        width = int(lengths.max())
        if width + max(0, max_tokens - 1) > self.max_length:
            raise ValueError("MagicPIG prompt plus decode steps exceeds max_length")
        eos_ids = self.eos_token_ids if eos_token_id is None else eos_token_id
        if eos_ids is None:
            eos_ids = []
        elif isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        else:
            eos_ids = list(eos_ids)
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_ids[0] if eos_ids else 0
        output = input_ids.new_full((batch, prefix_width + max_tokens), pad_token_id)
        output[:, :prefix_width] = input_ids
        sequence = input_ids.new_full((batch, width), pad_token_id)
        for row, length in enumerate(lengths.tolist()):
            sequence[row, -length:] = input_ids[row, valid[row]]
        mask = torch.arange(width, device=self.device)[None, :] >= (width - lengths[:, None])
        positions = (mask.long().cumsum(dim=-1) - 1).clamp_min(0)
        eos = torch.tensor(eos_ids, device=self.device, dtype=input_ids.dtype)
        finished = torch.zeros(batch, device=self.device, dtype=torch.bool)
        self.generate_token = 0
        self.prefill_latency = self.decode_latency = self.TPOT = 0.0
        state = self.attention_server
        state.reset()
        state.prompt_mask = mask
        past = outputs = logits = None
        try:
            if max_tokens:
                state.is_prefill = True
                self._synchronize()
                start = time.perf_counter()
                for begin in range(0, width, self.chunk_size):
                    end = min(begin + self.chunk_size, width)
                    outputs = self.model(
                        input_ids=sequence[:, begin:end], attention_mask=mask[:, :end],
                        position_ids=positions[:, begin:end], past_key_values=past,
                        use_cache=True, logits_to_keep=1,
                    )
                    past = self.past_key_values = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    outputs = None
                self._synchronize()
                self.prefill_latency = time.perf_counter() - start
                state.is_prefill = False
                start = time.perf_counter()
                for step in range(max_tokens):
                    next_tokens = logits.argmax(dim=-1)
                    if not self.fixed_output_length:
                        next_tokens = next_tokens.masked_fill(finished, pad_token_id)
                    output[:, prefix_width + step] = next_tokens
                    self.generate_token += 1
                    if not self.fixed_output_length and eos.numel():
                        finished |= (next_tokens[:, None] == eos[None, :]).any(dim=-1)
                        if finished.all():
                            break
                    if step + 1 < max_tokens:
                        mask = torch.cat((mask, torch.ones(batch, 1, device=self.device, dtype=mask.dtype)), dim=-1)
                        outputs = self.model(
                            input_ids=next_tokens[:, None], attention_mask=mask,
                            position_ids=lengths[:, None] + step, past_key_values=past,
                            use_cache=True, logits_to_keep=1,
                        )
                        past = self.past_key_values = outputs.past_key_values
                        logits = outputs.logits[:, -1, :]
                        outputs = None
                self._synchronize()
                self.decode_latency = time.perf_counter() - start
                self.TPOT = self.decode_latency / max(1, self.generate_token - 1) * 1000
            self.latency = self.Latency = self.lantency = self.prefill_latency + self.decode_latency
            self.sparse.append(state.sparse)
            self.sampled_tokens.append(state.sampled_tokens)
            self.decode_tokens.append(state.decode_tokens)
            output = output[:, :prefix_width + self.generate_token]
            return output[0].tolist() if batch == 1 else output
        finally:
            past = outputs = logits = None
            self.clear()
