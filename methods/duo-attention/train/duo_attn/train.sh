duo_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$duo_script_dir/.." || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [ "$#" -lt 1 ]; then
    printf 'Usage: bash %s <model_name_or_path> [training options]\n' "$0" >&2
    exit 2
fi
duo_model_name="$1"
shift

# Example usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash duo_attn/train.sh <model_name_or_path>
#
# One process per visible GPU; NPROC_PER_NODE can select fewer devices.
duo_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a duo_device_ids <<< "$duo_visible_devices"
duo_processes="${NPROC_PER_NODE:-${#duo_device_ids[@]}}"
if [[ ! "$duo_processes" =~ ^[1-9][0-9]*$ ]] || (( duo_processes > ${#duo_device_ids[@]} )); then
    printf 'NPROC_PER_NODE must be between 1 and the number of visible GPUs.\n' >&2
    exit 2
fi
CUDA_VISIBLE_DEVICES="$duo_visible_devices" python -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node "$duo_processes" -m duo_attn.train \
    --model_name "$duo_model_name" \
    --batch_size 1 \
    --max_length 32000  \
    --dataset_name duo_attn/datasets/booksum.jsonl.zst \
    --sink_size 128 \
    --recent_size 256 \
    --num_steps 2000 \
    --lr 0.02 \
    --reg_weight 0.05 \
    --exp_name "$(basename "$duo_model_name")/formula" \
    --min_needle_depth_ratio 0.05 \
    --max_needle_depth_ratio 0.95 \
    --context_length_min 1000 \
    --context_length_max 32000  \
    --context_lengths_num_intervals 50 \
    --gradient_accumulation_steps 1 \
    --num_passkeys 10 \
    --depth_ratio_num_intervals 1000 \
    --dataset_format multiple_passkey \
    --output_dir "duo_attn/attn_patterns/$(basename "$duo_model_name")" \
    --streaming_attn_implementation blocksparse \
    --activation_offload \
    --ffn_chunk_size 2048 \
    --disable_wandb "$@"
