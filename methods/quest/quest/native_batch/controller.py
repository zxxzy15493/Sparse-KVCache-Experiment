"""Batched Quest controller following the original begin/end-forward lifecycle."""

import torch

from .kv_cache import BatchedKvPool
from .ops import load_kernels


class NativeBatchController:
    def __init__(self, layers, batch_size, q_heads, kv_heads, dim, max_seq_len,
                 token_budget, page_size=16, device='cuda', legacy_head_selection=True):
        if min(layers, batch_size, q_heads, kv_heads, max_seq_len, page_size) <= 0:
            raise ValueError('Dimensions and capacities must be positive')
        if q_heads % kv_heads or dim not in (64, 128):
            raise ValueError('Invalid GQA ratio or native head dimension')
        if page_size not in (1, 8, 16, 32):
            raise ValueError('Native paged prefill requires page_size 1, 8, 16 or 32')
        self.page_budget = token_budget // page_size
        if self.page_budget < 1:
            raise ValueError('Token budget must cover at least one page')
        self.real_q_heads, self.kv_heads = q_heads, kv_heads
        group = q_heads // kv_heads
        groups = [g for g in (1, 4, 8) if g >= group]
        if not groups:
            raise ValueError('Native Quest supports GQA groups up to 8')
        self.q_heads = kv_heads * groups[0]
        self.dim, self.batch_size = dim, batch_size
        self.legacy_head_selection = legacy_head_selection
        self.pool = BatchedKvPool(layers, batch_size, kv_heads, dim, max_seq_len, page_size, device)
        self.kernels = load_kernels()
        self.plan = None
        self.plan_builds = 0
        self.layout_updates = 0
        self.query_length = 0

    def prepare(self, query_length, dense=False):
        self.end_forward()
        self.is_prefill = self.pool.seq_len == 0
        if not self.is_prefill and query_length != 1:
            raise ValueError('Decode requires one token per request; clear before a new prompt')
        self.position = self.pool.seq_len
        self.query_length = query_length
        self.pool.reserve(query_length)
        if self.is_prefill:
            self.query_indptr = torch.arange(self.batch_size + 1, dtype=torch.int32,
                                             device=self.pool.device) * query_length
        self.layout_updates += 1
        self.begin_forward(dense=dense)

    def begin_forward(self, dense=False, update_tensors=True):
        if self.plan is not None:
            raise RuntimeError('Call end_forward before preparing another decode phase')
        if self.is_prefill:
            return
        if update_tensors:
            self.full_indices = self.pool.page_ids[:, None, :-1].repeat(1, self.q_heads, 1)
        self.inference_page_budget = self.pool.pages if dense else min(self.pool.pages, self.page_budget)
        selected = self.inference_page_budget
        # Match the original controller: allocate and zero TopK buffers at each begin_forward.
        self.top_values = torch.zeros(self.batch_size, self.q_heads, selected - 1,
                                      device=self.pool.device, dtype=torch.float16)
        self.top_indices = torch.zeros_like(self.top_values, dtype=torch.int32)
        scratch_bytes = max(4096, self.batch_size * self.q_heads * self.pool.pages * 16)
        self.top_scratch = torch.zeros(scratch_bytes, device=self.pool.device, dtype=torch.uint8)
        self.decode_indptr = torch.arange(self.batch_size + 1, dtype=torch.int32,
                                         device=self.pool.device) * (selected - 1)
        with torch.cuda.device(self.pool.device):
            self.plan = self.kernels.DecodePlan(self.decode_indptr.cpu(), self.q_heads, self.kv_heads,
                self.dim, self.pool.page_size, not self.legacy_head_selection)
        self.virtual_indptr = torch.arange(self.batch_size + 1, dtype=torch.int32,
                                           device=self.pool.device) * selected
        self.plan_builds += 1

    def end_forward(self):
        if self.plan is not None:
            self.plan.close()
            self.plan = None

    def append(self, layer, k, v):
        self.kernels.append(k.contiguous(), v.contiguous(), self.pool.data[layer], self.pool.metadata[layer],
                            self.pool.indices, self.pool.metadata_indices, self.pool.indptr,
                            self.pool.metadata_indptr, self.pool.last_len, self.pool.metadata_last_len)

    def attention(self, layer, q, dense=False):
        if self.is_prefill:
            return self.kernels.prefill(q.contiguous(), self.pool.data[layer], self.pool.indices,
                                        self.pool.indptr, self.query_indptr, self.pool.last_len)
        selected = self.pool.pages if dense else min(self.page_budget, self.pool.pages)
        if self.plan is None or self.inference_page_budget != selected:
            self.end_forward()
            self.begin_forward(dense=dense, update_tensors=False)
        indices = self.full_indices
        query = q[:, 0].contiguous()
        if selected < self.pool.pages:
            torch.cuda.synchronize(self.pool.device)
            if selected > 1:
                self.scores = torch.empty(self.batch_size, self.q_heads, self.pool.pages - 1,
                                          device=self.pool.device, dtype=torch.float16)
                metadata = self.pool.metadata[layer]
                if self.q_heads != self.kv_heads:
                    metadata = metadata.repeat_interleave(self.q_heads // self.kv_heads, dim=3)
                self.kernels.estimate(query, self.scores, metadata, self.pool.metadata_indices,
                                      self.pool.metadata_indptr, self.pool.metadata_last_len)
                self.kernels.topk(self.scores, self.full_indices, self.top_values,
                                  self.top_indices, self.top_scratch)
            torch.cuda.synchronize(self.pool.device)
            indices = self.top_indices
        output = self.plan.run(query, self.pool.data[layer], indices, self.virtual_indptr,
                               self.pool.last_indices, self.pool.last_len)
        if selected < self.pool.pages:
            torch.cuda.synchronize(self.pool.device)
        return output[:, None]

    def clear(self):
        self.end_forward()
        self.pool.clear()
        for name in ('full_indices','scores','top_values','top_indices','top_scratch',
                     'query_indptr','decode_indptr','virtual_indptr','inference_page_budget'):
            if hasattr(self, name):
                delattr(self, name)
