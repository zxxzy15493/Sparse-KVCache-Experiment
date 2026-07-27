import torch
import time
import os
import numpy as np
import math
from datetime import datetime
import sys
def repeat(a:torch.Tensor, size, dim_idx):
    shape = a.shape
    return a.unsqueeze(dim_idx+1) \
            .expand(*shape[:dim_idx], shape[dim_idx], size, *shape[dim_idx+1:]) \
            .reshape(*shape[:dim_idx], shape[dim_idx] * size, *shape[dim_idx+1:])

def unrepeat(a:torch.Tensor, size, dim_idx):
    shape = a.shape
    return a.reshape(*shape[:dim_idx], shape[dim_idx] // size, size, *shape[dim_idx+1:]) \
            .select(dim_idx+1, 0) # NOTE: By default it will squeeze the target dimension.

_recall_file = None
_recall_prompt_id = 0

def write_recall_prompt_separator(prefill_len=None):
    global _recall_prompt_id

    f = get_recall_file()
    if f is None:
        return None

    _recall_prompt_id += 1
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # cmd = " ".join([sys.executable] + sys.argv)

    f.write("\n")
    f.write(f"# ===== PROMPT {_recall_prompt_id} | {now} =====\n")
    if prefill_len is not None:
        f.write(f"# prefill_len: {prefill_len}\n")
    # f.write(f"# command: {cmd}\n")
    f.flush()

    return _recall_prompt_id

def get_recall_file():
    global _recall_file

    if _recall_file is not None:
        return _recall_file

    recall_name = os.environ.get("RECALL_NAME", "recall")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = os.path.join("recall_list", recall_name)
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, f"{recall_name}_{now}.csv")

    _recall_file = open(filename, "a", buffering=1)
    _recall_file.write("layer,head,recall,recall@100,selected_attn\n")

    return _recall_file


def calc_recall(query, key, dummy_topk_indices, num_kv_group, topk_size, layer_idx=0):
   
    _, kv_head_num, kv_seq_len, dim = key.shape
    _, n_head, q_len, _ = query.shape

    if key.shape[1] * num_kv_group == query.shape[1]:
        real_weight = query.float() @ repeat(key, num_kv_group, 1).transpose(2,3).float()
    elif key.shape[1] == query.shape[1]:
        real_weight = query.float() @ key.transpose(2,3).float()
    else:
        raise Exception(f"?{key.shape},{query.shape},{num_kv_group}")

    attn_scale = math.sqrt(dim)
    real_attn_score = torch.softmax(real_weight / attn_scale, dim=-1).to(query.dtype)

    real_topk_indices = real_weight.topk(k=topk_size, dim=-1, largest=True).indices
    real_top100_indices = real_weight.topk(k=min(100, kv_seq_len), dim=-1, largest=True).indices

    if dummy_topk_indices.shape[1] != real_topk_indices.shape[1]:
        dummy_topk_indices = repeat(dummy_topk_indices, num_kv_group, 1)

    dummy_topk_indices = dummy_topk_indices.flatten(0,1)
    real_topk_indices = real_topk_indices.flatten(0,1)
    real_top100_indices = real_top100_indices.flatten(0,1)

    f = get_recall_file()
    if layer_idx == 0:
        f.write("\n")
    f.flush()
    for h in range(n_head):
        dummy = dummy_topk_indices[h, 0, :]
        real = real_topk_indices[h, 0, :]
        real100 = real_top100_indices[h, 0, :]

        assert dummy.numel() == torch.unique(dummy).numel()

        comparison = torch.isin(dummy, real, assume_unique=True)
        hit_cnt = torch.sum(comparison.int()).item()
        recall_rate = hit_cnt / topk_size

        comparison100 = torch.isin(dummy, real100, assume_unique=True)
        hit100_cnt = torch.sum(comparison100.int()).item()
        recall100_rate = hit100_cnt / min(100, kv_seq_len)

        head_real_attn = real_attn_score[0, h, 0, :]
        selected_attn = head_real_attn[dummy].sum().item()

        f.write(f"{layer_idx:3d},{h:3d},{recall_rate:7.4f},{recall100_rate:7.4f},{selected_attn:7.4f}\n")

    



class RetrievalBasedCompressor:
    def __init__(self, **kwargs) -> None:
        self.profile_metric = {
            "prefill_time" : 0,
            "prefill_attn_time" : 0,
            "prefill_cnt": 0,
            "prefill_per_layer_time": 0,

            "prepare_idx_elapsed" : 0,
            "prepare_idx_cnt" : 0,

            "decoding_time" : 0,
            "decoding_cnt" : 0,
            "decoding_attn_time" : 0,

            "offload_pq_ref_elapsed" : 0,
            "offload_kv_elapsed" : 0,
            "offload_cnt": 0,
            "offload_pq_ref_bytes" : 0,
            "offload_kv_bytes" : 0,

            "fetch_ref_elapsed" : 0,
            "fetch_kv_elapsed" : 0,
            "cpu_gather_elapsed" : 0,
            "calc_dummy_weight_elapsed" : 0,
            "fetch_ref_data_bytes" : 0,
            "fetch_kv_data_bytes" : 0,
        }

        self.device = kwargs["cur_device"]
    
    def profile_ckpt(self):
        torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def reset(self):
        for k,_ in self.profile_metric.items():
            self.profile_metric[k] = 0

    def showtime(self):
        result_str = "\n".join([f"{key} : {value}" for key, value in self.profile_metric.items()])
        print("-----profile result show:\n", result_str)
        with open(f"./profile_result/mistral_profile_{os.getpid()}","a") as f:
            f.write(result_str)
