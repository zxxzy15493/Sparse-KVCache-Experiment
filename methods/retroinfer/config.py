import os
import sys
import json
import math
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(PROJECT_ROOT)

def parse_attn_args(parser: argparse.ArgumentParser):
    parser.add_argument("--budget_ratio", type=float, default=0.018, help="ratio of budget")
    parser.add_argument("--estimate_ratio", type=float, default=0.25, help="ratio of estimated clusters for RetriveInfer")
    parser.add_argument("--budget", type=int, default=1024, help="the number of clusters to retrieve")
    parser.add_argument("--ratio_or_fixed", type=int, default=1, help="1: fixed budget, 0: ratio")

    return parser


def generate_config(
    model_name, 
    context_len, 
    attn_type,
    budget_ratio=0.018,
    budget=1024,
    estimate_ratio=0.25,
    # default retrieve infer configs
    n_segments=None,
    measure_vram : int = 0,
    RECALL: int = 0,
    ratio_or_fixed=1
):
    aprox_cluster_size = 16

    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    MODEL_NAME = model_name.split("/")[-1]+'.json'
    CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)
    with open(CONFIG_FILE, "r") as f:
        original_config = json.load(f)
    
    if n_segments is None:
        n_segments = max(1, context_len // 8192)
    
    n_clusters = math.ceil(context_len/aprox_cluster_size)

    if attn_type == 'RetroInfer':
        # compute the nearest multiple of (n_segments*32)
        lower = (n_clusters // (n_segments*32)) * (n_segments*32)
        upper = lower + (n_segments*32)
        n_clusters = lower if abs(n_clusters - lower) <= abs(n_clusters - upper) else upper
        if n_clusters < 32:
            n_clusters = 32
    
    
    # nprobe
    # estimatetoken，retrieval16token（16token）
    if ratio_or_fixed == 1:
        nprobe = min(max(1, int((budget - 16 - 32) / (aprox_cluster_size * 2))), n_clusters)
        estimate_size = min(max(1, (budget - 16 - 32 - nprobe * 16)), max(0, n_clusters - nprobe))
    elif ratio_or_fixed == 0:
        nprobe = max(1, int(n_clusters*budget_ratio))
        estimate_size = max(1, int(n_clusters*estimate_ratio))
    print(f"context_len: {context_len}, n_clusters: {n_clusters}, nprobe: {nprobe}, n_segments: {n_segments}")


    if attn_type == 'RetroInfer':
        # sinklocal
        original_config[attn_type]['static_pattern_start'] = 16
        original_config[attn_type]['static_pattern_end'] = 32

        original_config[attn_type]['n_centroids'] = n_clusters
        original_config[attn_type]['n_segment'] = n_segments
        original_config[attn_type]['nprobe'] = nprobe
        original_config[attn_type]['cache_cluster_num'] = int(nprobe*3)
        original_config[attn_type]['max_compute_cluster_num'] = estimate_size + nprobe
        print(original_config[attn_type])
    if attn_type == 'RetroInfer' and measure_vram == 1:
        original_config[attn_type]['static_pattern_start'] = 4
        original_config[attn_type]['static_pattern_end'] = 4
        print(original_config[attn_type])
    
    
    return original_config