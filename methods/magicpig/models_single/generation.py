"""Shared greedy generation: per-request prefill followed by batched decode."""

import time

import torch


def _synchronize(model):
    if torch.device(model.device).type == "cuda":
        torch.cuda.synchronize(model.device)


@torch.inference_mode()
def generate_batch(model, input_ids, max_tokens=128, attention_mask=None,
                   eos_token_id=None, pad_token_id=None):
    """Return a legacy flat list for B=1, or a [B, prompt+output] tensor.

    The batch must match the attention server's allocated batch size. Padding
    is removed for prefill, but the original padded prefix is kept in output.
    ``fixed_output_length`` disables EOS stopping for efficiency benchmarks.
    """
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
    if input_ids.ndim != 2 or not input_ids.shape[1]:
        raise ValueError("MagicPig input_ids must have shape [batch_size, nonempty sequence].")
    batch_size, prefix_width = input_ids.shape
    if batch_size != model.batch_size:
        raise ValueError(
            f"MagicPig was allocated for batch_size={model.batch_size}, "
            f"but received {batch_size} sequences. Load it with the requested batch_size."
        )
    if batch_size > 1 and getattr(model, "RECALL", False):
        raise ValueError("MagicPig RECALL mode currently requires batch_size=1.")
    input_ids = input_ids.to(model.device)
    if attention_mask is None:
        lengths = torch.full((batch_size,), prefix_width, device=model.device, dtype=torch.long)
        spans = [(0, prefix_width)] * batch_size
    elif attention_mask.shape != input_ids.shape:
        raise ValueError("MagicPig attention_mask must have the same shape as input_ids.")
    else:
        attention_mask = attention_mask.to(device=model.device, dtype=torch.bool)
        lengths = attention_mask.sum(dim=1)
        starts = attention_mask.int().argmax(dim=1)
        spans = torch.stack((starts, lengths), dim=1).tolist()
        if any(length == 0 for _, length in spans):
            raise ValueError("MagicPig cannot prefill an empty prompt.")
        # One contiguous run of valid tokens, including either padding side.
        runs = (attention_mask[:, 1:] != attention_mask[:, :-1]).sum(dim=1)
        runs = runs + attention_mask[:, 0] + attention_mask[:, -1]
        if not (runs == 2).all():
            raise ValueError("MagicPig attention_mask only supports contiguous left/right padding.")
    # These lengths also drive per-request rotary positions during decode.
    prompt_lengths = [length for _, length in spans]

    fixed_length = int(model.fixed_output_length)
    max_tokens = fixed_length if fixed_length else int(max_tokens)
    if max_tokens < 0:
        raise ValueError("MagicPig max_tokens must be nonnegative.")
    # The last output token does not need another forward/KV append.
    decode_steps = max(0, max_tokens - 1)
    if max(prompt_lengths) + decode_steps > model.max_length:
        raise ValueError("MagicPig prompt plus decode steps exceeds max_length.")
    server = model.attention_server
    if max_tokens and decode_steps > server.generation_buffer:
        raise ValueError(
            f"MagicPig needs generation_buffer >= {decode_steps}; "
            f"the allocated buffer is {server.generation_buffer}."
        )

    if eos_token_id is None:
        eos_token_id = getattr(model, "eos_token_ids", getattr(model.config, "eos_token_id", None))
    if eos_token_id is None:
        eos_ids = []
    elif isinstance(eos_token_id, int):
        eos_ids = [eos_token_id]
    else:
        eos_ids = list(eos_token_id)
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_ids[0] if eos_ids else 0
    eos = torch.tensor(eos_ids, device=model.device, dtype=input_ids.dtype)
    finished = torch.zeros(batch_size, device=model.device, dtype=torch.bool)
    output = input_ids.new_full((batch_size, prefix_width + max_tokens), pad_token_id)
    output[:, :prefix_width] = input_ids
    model.generate_token = 0
    model.prefill_latency = model.decode_latency = model.TPOT = 0.0

    try:
        if max_tokens:
            _synchronize(model)
            start = time.perf_counter()
            # Scratch KV/hash buffers are reused only after the previous
            # request has been copied into its own CPU and GPU cache slots.
            logits = torch.cat([
                model.prefill(input_ids[row:row + 1, begin:begin + length], request_id=row)
                for row, (begin, length) in enumerate(spans)
            ], dim=0)
            _synchronize(model)
            model.prefill_latency = time.perf_counter() - start
            start = time.perf_counter()
            positions = lengths[:, None]
            for step in range(max_tokens):
                next_tokens = logits.argmax(dim=-1).reshape(batch_size, 1)
                if not fixed_length:
                    next_tokens = next_tokens.masked_fill(finished[:, None], pad_token_id)
                output[:, prefix_width + step] = next_tokens[:, 0]
                model.generate_token += 1
                if not fixed_length and eos.numel():
                    finished |= (next_tokens == eos[None, :]).any(dim=1)
                    if finished.all():
                        break
                if step + 1 < max_tokens:
                    logits = model.inference(input_ids=next_tokens, position_ids=positions + step)
            _synchronize(model)
            model.decode_latency = time.perf_counter() - start
            model.TPOT = model.decode_latency / max(1, model.generate_token - 1) * 1000
        model.latency = model.Latency = model.lantency = model.prefill_latency + model.decode_latency
        output = output[:, :prefix_width + model.generate_token]
        return output[0].tolist() if batch_size == 1 else output
    finally:
        # A failed prefill/decode must not contaminate the next benchmark batch.
        model.clear()
