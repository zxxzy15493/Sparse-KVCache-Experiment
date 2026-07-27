import torch
import time

class TimeManager:
    def __init__(self):
        
        self.decode_step = 0
        ####################################### prefill ###########################################
        self.prefill_latency = 0

        self.layer_prefill_time = 0
        self.layer_prefill_event_time = 0
        self.layer_prefill_start_event = None
        self.layer_prefill_end_event = None

        self.prefill_preAttn_ffn_time = 0
        self.prefill_preAttn_ffn_event_time = 0
        self.prefill_preAttn_ffn_start_event = None
        self.prefill_preAttn_ffn_end_event = None

        self.prefill_postAttn_ffn_time = 0
        self.prefill_postAttn_ffn_event_time = 0
        self.prefill_postAttn_ffn_start_event = None
        self.prefill_postAttn_ffn_end_event = None

        self.prefill_attn_time = 0
        self.prefill_attn_event_time = 0
        self.prefill_attn_start_event = None
        self.prefill_attn_end_event = None

        self.cluster_time = 0
        self.cluster_event_time = 0
        self.cluster_start_event = None
        self.cluster_end_event = None

        self.sync_time = 0
        self.sync_event_time = 0
        self.sync_start_event = None
        self.sync_end_event = None

        ############################# KVCache prefill ##############################
        self.unload_time = 0
        self.unload_event_time = 0
        self.unload_start_event = None
        self.unload_end_event = None

        self.construct_time = 0
        self.construct_event_time = 0
        self.construct_start_event = None
        self.construct_end_event = None

        self.kvcache_sync_time = 0
        self.kvcache_sync_event_time = 0
        self.kvcache_sync_start_event = None
        self.kvcache_sync_end_event = None

        ####################################### decode ###########################################
        self.decode_latency = 0

        self.decode_attn_time = 0
        self.decode_attn_event_time = 0
        self.decode_attn_start_event = None
        self.decode_attn_end_event = None

        self.decode_preAttn_ffn_time = 0
        self.decode_preAttn_ffn_event_time = 0
        self.decode_preAttn_ffn_start_event = None
        self.decode_preAttn_ffn_end_event = None

        self.decode_postAttn_ffn_time = 0
        self.decode_postAttn_ffn_event_time = 0
        self.decode_postAttn_ffn_start_event = None
        self.decode_postAttn_ffn_end_event = None

        ###################################### kvcache decode #####################################
        self.compute_method_time = 0
        self.compute_method_event_time = 0
        self.compute_method_start_event = None
        self.compute_method_end_event = None

        self.retrieve_time = 0
        self.retrieve_event_time = 0
        self.retrieve_start_event = None
        self.retrieve_end_event = None

        self.load_time = 0
        self.load_event_time = 0
        self.load_start_event = None
        self.load_end_event = None

        self.esitimate_load_time = 0
        self.esitimate_load_event_time = 0
        self.esitimate_load_start_event = None
        self.esitimate_load_end_event = None

        self.esitimate_attn_time = 0
        self.esitimate_attn_event_time = 0
        self.esitimate_attn_start_event = None
        self.esitimate_attn_end_event = None

        self.attn_time = 0
        self.attn_event_time = 0
        self.attn_start_event = None
        self.attn_end_event = None

        self.update_time = 0
        self.update_event_time = 0
        self.update_start_event = None
        self.update_end_event = None

    def myinit(self, step, num_layers):
        self.step = step - 1
        self.num_layers = num_layers
        ####################################### prefill ###########################################
        self.prefill_latency = 0

        self.layer_prefill_time = 0
        self.layer_prefill_event_time = 0
        self.layer_prefill_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]
        self.layer_prefill_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]

        self.prefill_preAttn_ffn_time = 0
        self.prefill_preAttn_ffn_event_time = 0
        self.prefill_preAttn_ffn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]
        self.prefill_preAttn_ffn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]

        self.prefill_postAttn_ffn_time = 0
        self.prefill_postAttn_ffn_event_time = 0
        self.prefill_postAttn_ffn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]
        self.prefill_postAttn_ffn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]

        self.prefill_attn_time = 0
        self.prefill_attn_event_time = 0
        self.prefill_attn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]
        self.prefill_attn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]

        self.cluster_time = 0
        self.cluster_event_time = 0
        self.cluster_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]
        self.cluster_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]

        self.sync_time = 0
        self.sync_event_time = 0
        self.sync_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]
        self.sync_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]

        ############################# KVCache prefill ##############################
        self.unload_time = 0
        self.unload_event_time = 0
        self.unload_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]
        self.unload_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]


        # ，organize_kv，segment_kmeans()，kv_cache.prefill_update_kv_cache()segment_kmeans()
        self.construct_time = 0
        self.construct_event_time = 0
        self.construct_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]
        self.construct_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]

        # sync_time
        self.kvcache_sync_time = 0
        self.kvcache_sync_event_time = 0
        self.kvcache_sync_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]
        self.kvcache_sync_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers)]


        ####################################### decode ###########################################
        self.decode_latency = 0

        self.decode_attn_time = 0 
        self.decode_attn_event_time = 0
        self.decode_attn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]
        self.decode_attn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]


        self.decode_preAttn_ffn_time = 0
        self.decode_preAttn_ffn_event_time = 0
        self.decode_preAttn_ffn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]
        self.decode_preAttn_ffn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]

        self.decode_postAttn_ffn_time = 0
        self.decode_postAttn_ffn_event_time = 0
        self.decode_postAttn_ffn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]
        self.decode_postAttn_ffn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]

        ###################################### kvcache decode #####################################
        # compute_method_time 
        self.compute_method_time = 0
        self.compute_method_event_time = 0
        self.compute_method_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]
        self.compute_method_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]

        # ，kv_cache.compute()batch_gemm_softmax()topk()
        self.retrieve_time = 0
        self.retrieve_event_time = 0
        self.retrieve_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]
        self.retrieve_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]

        # ，kv_cache.compute() gather_copy_and_concat 
        self.load_time = 0
        self.load_event_time = 0
        self.load_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]
        self.load_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]

        # ，kv_cache.compute() gather_copy_vectors 
        self.esitimate_load_time = 0
        self.esitimate_load_event_time = 0
        self.esitimate_load_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]
        self.esitimate_load_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]

        # ，kv_cache.compute() weighted_flash_decoding 
        self.esitimate_attn_time = 0
        self.esitimate_attn_event_time = 0
        self.esitimate_attn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]
        self.esitimate_attn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]

        self.attn_time = 0
        self.attn_event_time = 0
        self.attn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]
        self.attn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]

        self.update_time = 0
        self.update_event_time = 0
        self.update_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]
        self.update_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_layers * self.step)]


    def get_time(self):
        return time.perf_counter()

    def get_final_time(self):
        torch.cuda.synchronize()

        # --- prefill per-layer events ---
        for layer in range(self.num_layers):
            self.layer_prefill_event_time += self.layer_prefill_start_event[layer].elapsed_time(self.layer_prefill_end_event[layer])
            self.prefill_preAttn_ffn_event_time += self.prefill_preAttn_ffn_start_event[layer].elapsed_time(self.prefill_preAttn_ffn_end_event[layer])
            self.prefill_postAttn_ffn_event_time += self.prefill_postAttn_ffn_start_event[layer].elapsed_time(self.prefill_postAttn_ffn_end_event[layer])
            self.prefill_attn_event_time += self.prefill_attn_start_event[layer].elapsed_time(self.prefill_attn_end_event[layer])
            self.cluster_event_time += self.cluster_start_event[layer].elapsed_time(self.cluster_end_event[layer])
            self.sync_event_time += self.sync_start_event[layer].elapsed_time(self.sync_end_event[layer])
            self.unload_event_time += self.unload_start_event[layer].elapsed_time(self.unload_end_event[layer])
            self.construct_event_time += self.construct_start_event[layer].elapsed_time(self.construct_end_event[layer])
            self.kvcache_sync_event_time += self.kvcache_sync_start_event[layer].elapsed_time(self.kvcache_sync_end_event[layer])

        # --- decode per-step-per-layer events ---
        for step in range(self.step):
            for layer in range(self.num_layers):
                idx = step * self.num_layers + layer
                self.decode_attn_event_time += self.decode_attn_start_event[idx].elapsed_time(self.decode_attn_end_event[idx])
                self.decode_preAttn_ffn_event_time += self.decode_preAttn_ffn_start_event[idx].elapsed_time(self.decode_preAttn_ffn_end_event[idx])
                self.decode_postAttn_ffn_event_time += self.decode_postAttn_ffn_start_event[idx].elapsed_time(self.decode_postAttn_ffn_end_event[idx])
                self.compute_method_event_time += self.compute_method_start_event[idx].elapsed_time(self.compute_method_end_event[idx])
                self.retrieve_event_time += self.retrieve_start_event[idx].elapsed_time(self.retrieve_end_event[idx])
                self.load_event_time += self.load_start_event[idx].elapsed_time(self.load_end_event[idx])
                self.esitimate_load_event_time += self.esitimate_load_start_event[idx].elapsed_time(self.esitimate_load_end_event[idx])
                self.esitimate_attn_event_time += self.esitimate_attn_start_event[idx].elapsed_time(self.esitimate_attn_end_event[idx])
                self.attn_event_time += self.attn_start_event[idx].elapsed_time(self.attn_end_event[idx])
                self.update_event_time += self.update_start_event[idx].elapsed_time(self.update_end_event[idx])

        prefill_ms = self.prefill_latency * 1000
        decode_ms_avg = self.decode_latency * 1000 / self.step

        time_info = {
            # --- prefill ---
            "prefill_latency": str(round(prefill_ms, 4)) + "ms",
            "layer_prefill_time": str(round(self.layer_prefill_time * 1000, 4)) + "ms",
            "layer_prefill_event_time": str(round(self.layer_prefill_event_time, 4)) + "ms",

            "prefill_attn_time": str(round(self.prefill_attn_time * 1000, 4)) + "ms",
            "prefill_attn_event_time": str(round(self.prefill_attn_event_time, 4)) + "ms",

            "prefill_preAttn_ffn_time": str(round(self.prefill_preAttn_ffn_time * 1000, 4)) + "ms",
            "prefill_preAttn_ffn_event_time": str(round(self.prefill_preAttn_ffn_event_time, 4)) + "ms",

            "prefill_postAttn_ffn_time": str(round(self.prefill_postAttn_ffn_time * 1000, 4)) + "ms",
            "prefill_postAttn_ffn_event_time": str(round(self.prefill_postAttn_ffn_event_time, 4)) + "ms",
            
            "cluster_time": str(round(self.cluster_time * 1000, 4)) + "ms",
            "cluster_event_time": str(round(self.cluster_event_time, 4)) + "ms",
            
            "sync_time": str(round(self.sync_time * 1000, 4)) + "ms",
            "sync_event_time": str(round(self.sync_event_time, 4)) + "ms",
            
            "unload_time": str(round(self.unload_time * 1000, 4)) + "ms",
            "unload_event_time": str(round(self.unload_event_time, 4)) + "ms",
            
            "construct_time": str(round(self.construct_time * 1000, 4)) + "ms",
            "construct_event_time": str(round(self.construct_event_time, 4)) + "ms",
            
            "kvcache_sync_time": str(round(self.kvcache_sync_time * 1000, 4)) + "ms",
            "kvcache_sync_event_time": str(round(self.kvcache_sync_event_time, 4)) + "ms",

            # --- decode (average per step) ---
            "decode_latency": str(round(decode_ms_avg, 4)) + "ms",
            "decode_attn_time": str(round(self.decode_attn_time * 1000 / self.step, 4)) + "ms",
            "decode_attn_event_time": str(round(self.decode_attn_event_time / self.step, 4)) + "ms",
            
            "decode_preAttn_ffn_time": str(round(self.decode_preAttn_ffn_time * 1000 / self.step, 4)) + "ms",
            "decode_preAttn_ffn_event_time": str(round(self.decode_preAttn_ffn_event_time / self.step, 4)) + "ms",
            
            "decode_postAttn_ffn_time": str(round(self.decode_postAttn_ffn_time * 1000 / self.step, 4)) + "ms",
            "decode_postAttn_ffn_event_time": str(round(self.decode_postAttn_ffn_event_time / self.step, 4)) + "ms",
            
            "ffn_time_1": str(round((self.decode_preAttn_ffn_event_time + self.decode_postAttn_ffn_event_time) / self.step, 4)) + "ms",
            "ffn_time_2": str(round((self.decode_latency * 1000 - self.compute_method_event_time) / self.step, 4)) + "ms",

            "compute_method_time": str(round(self.compute_method_time * 1000 / self.step, 4)) + "ms",
            "compute_method_event_time": str(round(self.compute_method_event_time / self.step, 4)) + "ms",
            
            "retrieve_time": str(round(self.retrieve_time * 1000 / self.step, 4)) + "ms",
            "retrieve_event_time": str(round(self.retrieve_event_time / self.step, 4)) + "ms",
            
            "load_time": str(round(self.load_time * 1000 / self.step, 4)) + "ms",
            "load_event_time": str(round(self.load_event_time / self.step, 4)) + "ms",
            
            "esitimate_load_time": str(round(self.esitimate_load_time * 1000 / self.step, 4)) + "ms",
            "esitimate_load_event_time": str(round(self.esitimate_load_event_time / self.step, 4)) + "ms",
            
            "esitimate_attn_time": str(round(self.esitimate_attn_time * 1000 / self.step, 4)) + "ms",
            "esitimate_attn_event_time": str(round(self.esitimate_attn_event_time / self.step, 4)) + "ms",
            
            "attn_time": str(round(self.attn_time * 1000 / self.step, 4)) + "ms",
            "attn_event_time": str(round(self.attn_event_time / self.step, 4)) + "ms",
            
            "update_time": str(round(self.update_time * 1000 / self.step, 4)) + "ms",
            "update_event_time": str(round(self.update_event_time / self.step, 4)) + "ms",
        }
        self.clear()
        return time_info
    
    def clear(self):
        self.decode_step = 0

        self.prefill_latency = 0
        self.layer_prefill_time = 0
        self.layer_prefill_event_time = 0
        self.prefill_attn_time = 0
        self.prefill_attn_event_time = 0
        self.prefill_preAttn_ffn_time = 0
        self.prefill_preAttn_ffn_event_time = 0
        self.prefill_postAttn_ffn_time = 0
        self.prefill_postAttn_ffn_event_time = 0
        self.cluster_time = 0
        self.cluster_event_time = 0
        self.sync_time = 0
        self.sync_event_time = 0
        self.unload_time = 0
        self.unload_event_time = 0
        self.construct_time = 0
        self.construct_event_time = 0
        self.kvcache_sync_time = 0
        self.kvcache_sync_event_time = 0

        self.decode_latency = 0
        self.decode_attn_time = 0
        self.decode_attn_event_time = 0
        self.decode_preAttn_ffn_time = 0
        self.decode_preAttn_ffn_event_time = 0
        self.decode_postAttn_ffn_time = 0
        self.decode_postAttn_ffn_event_time = 0
        self.compute_method_time = 0
        self.compute_method_event_time = 0
        self.retrieve_time = 0
        self.retrieve_event_time = 0
        self.load_time = 0
        self.load_event_time = 0
        self.esitimate_load_time = 0
        self.esitimate_load_event_time = 0
        self.esitimate_attn_time = 0
        self.esitimate_attn_event_time = 0
        self.attn_time = 0
        self.attn_event_time = 0
        self.update_time = 0
        self.update_event_time = 0


timeManager = TimeManager()