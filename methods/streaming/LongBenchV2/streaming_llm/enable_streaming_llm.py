from streaming_llm.kv_cache import StartRecentKVCache


def enable_streaming_llm(model, args):
    model.config.start_size = args.start_size
    model.config.recent_size = args.recent_size
    model.config.cache_size = args.start_size + args.recent_size
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_llama import (
            enable_llama_pos_shift_attention,
        )

        enable_llama_pos_shift_attention(model)
    elif "qwen" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_qwen import enable_qwen_pos_shift_attention
        enable_qwen_pos_shift_attention(model)
    elif "glm" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
        from streaming_llm.pos_shift.modify_glm import enable_glm_pos_shift_attention
        enable_glm_pos_shift_attention(model)
    else:
        raise ValueError(f"got {model.config.model_type}")

