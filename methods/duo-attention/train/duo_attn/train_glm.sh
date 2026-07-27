cd "$(dirname "$0")/../.."
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nnodes 1 --nproc_per_node 1 -m duo_attn.train \
    --model_name "zai-org/glm-4-9b-chat-1m" \
    --batch_size 1 \
    --max_length 131072 \
    --dataset_name duo_attn/Long-Data-Collections/pretrain/pile_sub.jsonl.zst \
    --sink_size 128 \
    --recent_size 256 \
    --num_steps 2000 \
    --lr 0.02 \
    --reg_weight 0.05 \
    --exp_name glm-4-9b-chat-1m/pile_sub-formal-8192-safe \
    --min_needle_depth_ratio 0.05 \
    --max_needle_depth_ratio 0.95 \
    --context_length_min 1000 \
    --context_length_max 131072 \
    --context_lengths_num_intervals 50 \
    --gradient_accumulation_steps 1 \
    --num_passkey 5 \
    --dataset_format multiple_passkey \
    --streaming_attn_implementation sdpa \
    --output_dir duo_attn/attn_patterns/glm-4-9b-chat-1m/pile_sub-safe \
    --disable_wandb \
    --initial_value 0.5