"""All-layer KV pools with separate physical pages for every request."""

import torch


class BatchedKvPool:
    def __init__(self, layers, batch_size, heads, dim, max_seq_len, page_size, device):
        self.batch_size, self.heads, self.dim = batch_size, heads, dim
        self.max_seq_len, self.page_size = max_seq_len, page_size
        self.max_pages = (max_seq_len + page_size - 1) // page_size
        self.max_metadata_pages = (self.max_pages + page_size - 1) // page_size
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Native Quest KV pools require a CUDA device')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        options = dict(dtype=torch.float16, device=self.device)
        self.data = torch.empty(layers, batch_size * self.max_pages, 2, page_size, heads, dim, **options)
        self.metadata = torch.empty(layers, batch_size * self.max_metadata_pages, 2, page_size, heads, dim, **options)
        self.seq_len = 0
        self.pages = 0
        self.metadata_pages = 0

    @property
    def last_len(self):
        return (self.seq_len - 1) % self.page_size + 1

    @property
    def metadata_last_len(self):
        return (self.pages - 1) % self.page_size + 1

    def reserve(self, length):
        if length <= 0 or self.seq_len + length > self.max_seq_len:
            raise ValueError('KV append must be nonempty and within capacity')
        self.seq_len += length
        pages = (self.seq_len + self.page_size - 1) // self.page_size
        changed = pages != self.pages
        self.pages = pages
        self.metadata_pages = (pages + self.page_size - 1) // self.page_size
        # Rebuild the layout tensors on every forward, as in the original Quest controller.
        options = dict(dtype=torch.int32, device=self.device)
        request = torch.arange(self.batch_size, **options)
        self.page_ids = (request[:, None] * self.max_pages + torch.arange(pages, **options)).contiguous()
        self.indices = self.page_ids.flatten()
        self.indptr = torch.arange(self.batch_size + 1, **options) * pages
        self.last_indices = self.page_ids[:, -1].contiguous()
        self.metadata_indices = (request[:, None] * self.max_metadata_pages +
                                 torch.arange(self.metadata_pages, **options)).flatten()
        self.metadata_indptr = torch.arange(self.batch_size + 1, **options) * self.metadata_pages
        return changed

    def clear(self):
        self.seq_len, self.pages, self.metadata_pages = 0, 0, 0
