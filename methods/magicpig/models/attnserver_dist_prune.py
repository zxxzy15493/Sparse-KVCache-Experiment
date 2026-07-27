from transformers import PretrainedConfig
import torch
try:
    from lsh import LSH
except ImportError as e:
    print(f"Failed to import LSH: {e}")
    LSH = None
try:
    from sparse_attention_cpu import SparseAttentionServer
except ImportError as e:
    print(f"Failed to import SparseAttentionServer: {e}")
    SparseAttentionServer = None
import flashinfer
import torch.distributed as dist

from termcolor import colored

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LSHSparseAttnServer:
    def __init__(self, 
        config :PretrainedConfig,
        K: int = 10,
        L: int = 150,
        batch_size :int = 1,
        num_sink_tokens :int = 16,
        num_local_tokens :int = 32,
        generation_buffer :int = 256,
        max_length: int = 8192,
        dense_layers: list[int] = [40], 
        device :str = 'cuda:0',
        dtype = torch.bfloat16,
        use_tensor_cores : bool = False,
        RECALL : bool = False
        ) -> None:
        
        self.RECALL = RECALL
        
        self.avg_nnz = 0
        self.count_nnz = 0
        
        self.K = K
        self.L = L
        self.config = config
        self.length = num_sink_tokens + num_local_tokens + generation_buffer
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers
        self.batch_size = batch_size
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.num_key_value_heads = config.num_key_value_heads // self.world_size
        self.num_attention_heads = config.num_attention_heads // self.world_size
        self.hidden_size = config.hidden_size // self.world_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dense_layers = dense_layers
        self.num_sink_tokens = num_sink_tokens
        self.num_local_tokens = num_local_tokens
        

        # for fixd budget
        self.hits_lsh_cpu = torch.zeros((self.batch_size * self.num_attention_heads, self.max_length)).to(torch.int32)
        # end

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
         
        self.avg_k = [torch.zeros(
            self.batch_size,
            self.num_key_value_heads,
            1,
            self.head_dim,
            device=self.device,
            dtype=self.dtype
        ) for _ in range(self.num_layers)] 
        
        self.attn_server = SparseAttentionServer()
        self.attn_server.alloc(self.num_layers, self.num_attention_heads, self.num_key_value_heads, self.head_dim, batch_size, max_length)
        self.lsh_retriever = LSH()
        self.lsh_retriever.alloc(self.K, self.L, self.num_layers, self.num_attention_heads, self.num_key_value_heads, batch_size, max_length)
        self.hash_func = torch.randn((self.head_dim, self.K * self.L), device=self.device, dtype=self.dtype)
        dist.broadcast(self.hash_func, 0)
        self.binary_pack = [int(2**i) for i in range(self.K)]
        self.binary_pack = torch.Tensor(self.binary_pack).to(device=self.device, dtype=torch.float16)
        
        self.nnz = torch.zeros((self.batch_size * self.num_attention_heads,)).to(torch.int32)
        self.results_lsh_cpu = torch.zeros((self.batch_size * self.num_attention_heads, self.max_length)).to(torch.int32)
        self.max_value_expsum = torch.ones((2, self.batch_size * self.num_attention_heads)).to(torch.float32).pin_memory()
        self.output_cuda = torch.zeros((self.batch_size * self.num_attention_heads, self.head_dim), dtype=torch.bfloat16).to(self.device)
        self.max_value_expsum_cuda = torch.ones((self.batch_size * self.num_attention_heads)).to(torch.float32).to(self.device)
        self.output = torch.zeros((self.batch_size * self.num_attention_heads, self.head_dim), dtype=torch.bfloat16).pin_memory()
        self.pinned_hashcode = torch.zeros((self.batch_size * self.num_attention_heads, self.L), dtype=torch.int32).pin_memory()
        self.pinned_query = torch.zeros((self.batch_size * self.num_attention_heads, self.head_dim), dtype=torch.bfloat16).pin_memory()
        self.chunk_size = 8192

        self.hash_code_buffer =  torch.zeros((self.num_key_value_heads, self.L, max_length), dtype=torch.int16, device=self.device)
        self.hash_code_buffer_cpu :torch.Tensor = None
        self.sorted_hash_values_buffer :torch.Tensor = None
        self.sorted_hash_indices_buffer :torch.Tensor = None
        
        self.max_num_pages = self.batch_size
        # self.length = num_sink_tokens + num_local_tokens + generation_buffer
        self.page_size = self.length
        self.kv_page_indices = torch.arange(self.max_num_pages).int().to(self.device)
        self.kv_page_indptr = torch.arange(self.batch_size + 1).int().to(self.device)
        self.kv_last_page_len = torch.zeros(self.batch_size).int().to(self.device)
        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        self.workspace_buffer, "HND", use_tensor_cores=use_tensor_cores
        )
        
        self.dense_max_num_pages = self.batch_size
        self.dense_page_size = self.max_length
        self.dense_kv_page_indices = torch.arange(self.dense_max_num_pages).int().to(self.device)
        self.dense_kv_page_indptr = torch.arange(self.batch_size + 1).int().to(self.device)
        self.dense_kv_last_page_len = torch.zeros(self.batch_size).int().to(self.device)
        self.batch_indices = torch.arange(self.batch_size).int().to(self.device)
        self.dense_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.dense_decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        self.dense_workspace_buffer, "HND", use_tensor_cores=use_tensor_cores
        )
          
        # shape --> (num_layers, max_num_pages, key/value, num_key_value_heads, page_size, head_dim)
        # max_num_pages == batch_size
        self.flashinfer_kv_cache = [
        torch.zeros(
            self.max_num_pages, 
            2, 
            self.num_key_value_heads, 
            self.dense_page_size if i in self.dense_layers else self.page_size, 
            self.head_dim, 
            dtype=torch.bfloat16, 
            device=self.device
        ) for i in range(self.num_layers)
        ]

    def alloc_buffer(self, seq_len:int):
        self.sorted_hash_values_buffer =  torch.zeros((self.num_key_value_heads, self.L, seq_len - self.num_sink_tokens - self.num_local_tokens), dtype=torch.int16, device="cpu").pin_memory()
        self.sorted_hash_indices_buffer =  torch.zeros((self.num_key_value_heads, self.L, seq_len - self.num_sink_tokens - self.num_local_tokens), dtype=torch.int32, device="cpu").pin_memory()
        
    def fill(self, 
        layer_idx:int,
        request_id: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_len:int):
        
         
        if layer_idx in self.dense_layers:
            self.flashinfer_kv_cache[layer_idx][request_id][0].copy_(key_cache.transpose(0,1))
            self.flashinfer_kv_cache[layer_idx][request_id][1].copy_(value_cache.transpose(0,1))
            self.dense_kv_last_page_len[request_id] = seq_len
        else:
           
            sink_tokens_key = key_cache[:self.num_sink_tokens]
            sink_tokens_value = value_cache[:self.num_sink_tokens]
            
            # contextnum_local_tokens
            local_tokens_key = key_cache[seq_len-self.num_local_tokens:seq_len]
            local_tokens_value = value_cache[seq_len-self.num_local_tokens:seq_len]
            
            key = torch.cat([sink_tokens_key, local_tokens_key], dim=0).transpose(0,1)
            value = torch.cat([sink_tokens_value, local_tokens_value], dim=0).transpose(0,1)
            
            # tokensCPU
            offload_key = key_cache[self.num_sink_tokens: seq_len-self.num_local_tokens]
            offload_value = value_cache[self.num_sink_tokens: seq_len-self.num_local_tokens]
            
            # offload_key.shape --> (head_num,  offload_len, head_dim)
            offload_key = offload_key.transpose(0,1).contiguous()
            offload_value = offload_value.transpose(0,1).contiguous()
            
            # Key，
            avg_k = offload_key.mean(dim=1, keepdim=True)
            
            key = key - avg_k
            offload_key = offload_key - avg_k
           
            kn = offload_key.norm(p=2, dim=-1).float()
            
            self.avg_k[layer_idx][request_id] = avg_k
            
            self.flashinfer_kv_cache[layer_idx][request_id][0][...,:self.num_sink_tokens + self.num_local_tokens,:].copy_(key)
            self.flashinfer_kv_cache[layer_idx][request_id][1][...,:self.num_sink_tokens + self.num_local_tokens,:].copy_(value)
            
            self.kv_last_page_len[request_id] = (self.num_sink_tokens + self.num_local_tokens)
            
            #shape --> (head_num, seq_len - num_sink - num_local, head_dim)
            offload_len = offload_key.shape[1]
            #  offload_key 
            num_iter = (offload_len // self.chunk_size) if (not offload_len % self.chunk_size) else (offload_len // self.chunk_size + 1)
            # hash_code
            
            # ，keyhashcode
            for i in range(num_iter):
                start = i * self.chunk_size
                end = min((i+1) * self.chunk_size, offload_len)
                #[head_num, offload_len, head_dim] @ [head_dim, K*L] --> [head_num, offload_len, K*L]
                hash_code = torch.matmul(offload_key[:,start:end,:], self.hash_func)
                hash_code = hash_code > 0
                # [head_num, offload_len, K*L] --> [head_num * offload_len * L, K]
                hash_code = hash_code.reshape(-1, self.K).to(torch.float16)
                # [head_num * offload_len * L, K] @ [K] --> [head_num * offload_len * L]
                hash_code = torch.mv(hash_code, self.binary_pack)
                # [head_num, offload_len, L]
                hash_code = hash_code.reshape(self.num_key_value_heads, -1, self.L)
                # [head_num, L, offload_len]  hash_code_buffer 
                hash_code = hash_code.transpose(1,2).contiguous().to(torch.int16)
                self.hash_code_buffer[:,:,start:end].copy_(hash_code)

            offload_key = offload_key.cpu()
            offload_value = offload_value.cpu()
            # offload_key  L2 CPU
            kn = kn.cpu()

            self.attn_server.fill(layer_idx, request_id, offload_key, offload_value, kn)
            
    def build_table(self, 
        layer_idx:int,
        request_id: int,
        seq_len:int):
        
        if layer_idx not in self.dense_layers:
            offload_len = seq_len - self.num_sink_tokens - self.num_local_tokens
            # head
            for i in range(self.num_key_value_heads):
                # hash_code_buffer shape --> (num_key_value_heads, L, offload_len)
                sorted_hash_values, sorted_hash_indices = self.hash_code_buffer[i,:,:offload_len].sort()
                # sorted_hash_values_buffer shape --> (num_key_value_heads, L, seq_len - num_sink - num_local)
                self.sorted_hash_values_buffer[i].copy_(sorted_hash_values)
                self.sorted_hash_indices_buffer[i].copy_(sorted_hash_indices)
                
            self.lsh_retriever.fill(layer_idx, request_id, 
                self.sorted_hash_values_buffer, 
                self.sorted_hash_indices_buffer)
        
    def plan(self):
        #  KV （+1， Token )
        self.kv_last_page_len += 1
        self.decode_wrapper.plan(
        self.kv_page_indptr,
        self.kv_page_indices,
        self.kv_last_page_len,
        self.num_attention_heads,
        self.num_key_value_heads,
        self.head_dim,
        self.page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        data_type=torch.bfloat16
    )     
        #  KV （+1， Token )
        self.dense_kv_last_page_len += 1
        self.dense_decode_wrapper.plan(
        self.dense_kv_page_indptr,
        self.dense_kv_page_indices,
        self.dense_kv_last_page_len,
        self.num_attention_heads,
        self.num_key_value_heads,
        self.head_dim,
        self.dense_page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        data_type=torch.bfloat16
    )     
    
    def decode(
        self, 
        query_states:torch.Tensor, 
        key_states:torch.Tensor, 
        value_states:torch.Tensor,
        layer_idx:int):
        
        if layer_idx in self.dense_layers:
            # dense_layers：key/valueflashinfer_kv_cache
            key_states = key_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            value_states = value_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            
            flashinfer.append_paged_kv_cache(
                key_states,
                value_states,
                self.batch_indices,
                self.dense_kv_last_page_len - 1,
                self.flashinfer_kv_cache[layer_idx],
                self.dense_kv_page_indices,
                self.dense_kv_page_indptr,
                self.dense_kv_last_page_len,
                kv_layout="HND"
            )
            q = query_states.reshape(self.batch_size, self.num_attention_heads, self.head_dim)
            hidden_states = self.dense_decode_wrapper.run(
            q, 
            self.flashinfer_kv_cache[layer_idx]
            )
            
            hidden_states = hidden_states.reshape(self.batch_size, 1, self.hidden_size)
           
            return hidden_states     
        else:
            bsz, _, q_len, _ = query_states.shape

            norm_q = query_states.reshape(-1, self.head_dim)
            norm_q = norm_q / norm_q.norm(p=2, dim=-1, keepdim=True) 
            q_hashcode = torch.matmul(norm_q, self.hash_func).gt(0)
            q_hashcode = q_hashcode.reshape(-1, self.K).to(torch.float16)
            q_hashcode = torch.mv(q_hashcode, self.binary_pack).int()
            q_hashcode = q_hashcode.reshape(self.batch_size * self.num_attention_heads, self.L)
            
            self.pinned_hashcode.copy_(q_hashcode)
            self.pinned_query.copy_(query_states.reshape(self.batch_size * self.num_attention_heads, self.head_dim))
            
            key_states = key_states - self.avg_k[layer_idx]
            
            key_states = key_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            value_states = value_states.reshape(self.batch_size, self.num_key_value_heads, self.head_dim)
            
            flashinfer.append_paged_kv_cache(
                key_states,
                value_states,
                self.batch_indices,
                self.kv_last_page_len - 1,
                self.flashinfer_kv_cache[layer_idx],
                self.kv_page_indices,
                self.kv_page_indptr,
                self.kv_last_page_len,
                kv_layout="HND"
            )
            
            q = query_states.reshape(self.batch_size, self.num_attention_heads, self.head_dim)
            gpu_hidden_states, gpu_lse = self.decode_wrapper.run_return_lse(
            q, 
            self.flashinfer_kv_cache[layer_idx]
            )

            self.lsh_retriever.batch_retrieve_count(layer_idx, self.pinned_hashcode, self.results_lsh_cpu, self.nnz, self.hits_lsh_cpu)
            
            
            self.avg_nnz += (self.nnz.sum() / self.num_attention_heads)
            self.count_nnz += 1

            self.attn_server.attention(layer_idx, self.K, self.L, self.output, self.max_value_expsum, self.pinned_query.float(), self.pinned_query.float().norm(p=2, dim=-1), self.results_lsh_cpu, self.nnz)
            
            self.max_value_expsum_cuda.copy_(self.max_value_expsum[1], non_blocking=True)
            self.output_cuda.copy_(self.output, non_blocking=True)

            cpu_lse = self.max_value_expsum_cuda.reshape(self.batch_size, self.num_attention_heads)
            
            cpu_hidden_states = self.output_cuda.reshape(self.batch_size, self.num_attention_heads, self.head_dim)
            hidden_states, _ = flashinfer.merge_state(gpu_hidden_states, gpu_lse, cpu_hidden_states, cpu_lse)
            

            hidden_states = hidden_states.reshape(bsz, q_len, self.hidden_size)
 
            return hidden_states 
            
    def clear(self):
        self.nnz.zero_()
        self.results_lsh_cpu.zero_()
        self.max_value_expsum.zero_()
        self.output_cuda.zero_()
        self.max_value_expsum_cuda.zero_()
        self.output.zero_()
        self.pinned_hashcode.zero_()
        self.pinned_query.zero_()
        for i in range(self.num_layers):
            self.avg_k[i].zero_()
            self.flashinfer_kv_cache[i].zero_()
            
        self.kv_last_page_len.zero_()
        self.dense_kv_last_page_len.zero_()
        
        self.lsh_retriever.clear()
        self.attn_server.clear()
