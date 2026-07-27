import torch


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    def __init__(
        self,
        start_size=16,
        recent_size=1008,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim

    def evict_for_space(self, key_states, value_states):
        seq_len = key_states.shape[self.k_seq_dim]
        
        if seq_len > self.cache_size:
            k_sink = key_states[:, :, :self.start_size, :]
            v_sink = value_states[:, :, :self.start_size, :]
            
            k_recent = key_states[:, :,  - self.recent_size:, :]
            v_recent = value_states[:, :,  - self.recent_size:, :]
            
            key_states = torch.cat([k_sink, k_recent], dim=self.k_seq_dim)
            value_states = torch.cat([v_sink, v_recent], dim=self.v_seq_dim)
            
        return key_states, value_states
   