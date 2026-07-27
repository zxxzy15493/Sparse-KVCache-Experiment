cd "$(dirname "$0")/.."
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"

# Example usage:
#   CUDA_VISIBLE_DEVICES=0 bash duo_attn/train.sh <model_name_or_path>
#
# Basic single-GPU run:
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nnodes 1 --nproc_per_node 1 -m duo_attn.train \
    --model_name "$1" \
    --batch_size 1 \
    --max_length 131072 \
    --dataset_name duo_attn/Long-Data-Collections/pretrain/pile_sub.jsonl.zst \
    --sink_size 128 \
    --recent_size 256 \
    --num_steps 2000 \
    --lr 0.02 \
    --reg_weight 0.05 \
    --exp_name "$(basename $1)/formula" \
    --min_needle_depth_ratio 0.05 \
    --max_needle_depth_ratio 0.95 \
    --context_length_min 1000 \
    --context_length_max 131072 \
    --context_lengths_num_intervals 50 \
    --gradient_accumulation_steps 1 \
    --num_passkey 5 \
    --dataset_format multiple_passkey \
    --output_dir duo_attn/attn_patterns/$(basename $1) \
    --streaming_attn_implementation sdpa \
    --disable_wandb