import torch
import time

class TimeManager:
    def __init__(self):
        self.decode_step = 0

        self.step = 0
        self.num_layers = 0
        self.num_heads = 0

        self.prefill_latency = 0

        self.pattern_allocate_time_head = 0
        self.pattern_allocation_event_time_head = 0
        self.pattern_allocation_start_event_head = None
        self.pattern_allocation_end_event_head = None

        self.decode_latency = 0

        self.prefill_attn_event_time = 0
        self.decode_attn_event_time = 0
        self.forward_attn_start_event = None
        self.forward_attn_end_event = None

        self.prefill_write_kv_event_time = 0
        self.decode_write_kv_event_time = 0
        self.write_kv_start_event = None
        self.write_kv_end_event = None

    def myinit(self, step, num_layers):
        self.decode_step = 0
        self.step = step
        self.num_layers = num_layers
        self.num_heads = num_layers

        self.prefill_latency = 0

        self.pattern_allocation_event_time_head = 0 
        self.pattern_allocation_start_event_head = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_heads * self.num_heads)]
        self.pattern_allocation_end_event_head = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_heads * self.num_heads)]

        self.decode_latency = 0

        self.prefill_attn_event_time = 0
        self.decode_attn_event_time = 0
        self.forward_attn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]
        self.forward_attn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]

        self.prefill_write_kv_event_time = 0
        self.decode_write_kv_event_time = 0
        self.write_kv_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]
        self.write_kv_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]


        self.prefill_pre_ffn_event_time = 0
        self.decode_pre_ffn_event_time = 0
        self.forward_pre_ffn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]
        self.forward_pre_ffn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]

        self.prefill_post_ffn_event_time = 0
        self.decode_post_ffn_event_time = 0
        self.forward_post_ffn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]
        self.forward_post_ffn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]

    def get_time(self):
        return time.perf_counter()

    def get_final_time(self, prefill_latency, decode_latency):
        num_heads = self.num_layers
        torch.cuda.synchronize()
        for layer in range(self.num_layers):
            for head in range(num_heads):
                self.pattern_allocation_event_time_head += self.pattern_allocation_start_event_head[layer * self.num_heads + head].elapsed_time(self.pattern_allocation_end_event_head[layer * self.num_heads + head])

        for layer in range(self.num_layers):
            self.prefill_attn_event_time += self.forward_attn_start_event[layer].elapsed_time(self.forward_attn_end_event[layer])
            self.prefill_write_kv_event_time += self.write_kv_start_event[layer].elapsed_time(self.write_kv_end_event[layer])

        for layer in range(self.num_layers):
            self.prefill_pre_ffn_event_time += self.forward_pre_ffn_start_event[layer].elapsed_time(self.forward_pre_ffn_end_event[layer])
            self.prefill_post_ffn_event_time += self.forward_post_ffn_start_event[layer].elapsed_time(self.forward_post_ffn_end_event[layer])

        for step in range(1, self.decode_step):
            for layer in range(self.num_layers):
                self.decode_attn_event_time += self.forward_attn_start_event[step * self.num_heads + layer].elapsed_time(self.forward_attn_end_event[step * self.num_heads + layer])
                self.decode_write_kv_event_time += self.write_kv_start_event[step * self.num_heads + layer].elapsed_time(self.write_kv_end_event[step * self.num_heads + layer])
                self.decode_pre_ffn_event_time += self.forward_pre_ffn_start_event[step * self.num_heads + layer].elapsed_time(self.forward_pre_ffn_end_event[step * self.num_heads + layer])
                self.decode_post_ffn_event_time += self.forward_post_ffn_start_event[step * self.num_heads + layer].elapsed_time(self.forward_post_ffn_end_event[step * self.num_heads + layer])

        time_info = {
            "prefill_latency": str(round(prefill_latency * 1000, 4)) + "ms",
            "pattern_allocation_event_time_head": str(round(self.pattern_allocation_event_time_head, 4)) + "ms",
            "prefill_write_kv_event_time": str(round(self.prefill_write_kv_event_time, 4)) + "ms",
            "prefill_attn_event_time": str(round(self.prefill_attn_event_time - self.pattern_allocation_event_time_head, 4)) + "ms",
            "prefill_ffn_event_time": str(round((prefill_latency * 1000 - self.prefill_attn_event_time - self.prefill_write_kv_event_time), 4)) + "ms",
            "prefill_pre_ffn_event_time": str(round(self.prefill_pre_ffn_event_time, 4)) + "ms",
            "prefill_post_ffn_event_time": str(round(self.prefill_post_ffn_event_time, 4)) + "ms",

            "decode_latency": str(round(decode_latency * 1000 / (self.step-1), 4)) + "ms",
            "decode_attn_event_time": str(round(self.decode_attn_event_time / (self.step-1), 4)) + "ms",
            "decode_write_kv_event_time": str(round(self.decode_write_kv_event_time / (self.step-1), 4)) + "ms",
            "decode_ffn_event_time": str(round((decode_latency * 1000 / (self.step-1) - self.decode_attn_event_time / (self.step-1) - self.decode_write_kv_event_time / (self.step-1)), 4)) + "ms",
            "decode_pre_ffn_event_time": str(round(self.decode_pre_ffn_event_time / (self.step-1), 4)) + "ms",
            "decode_post_ffn_event_time": str(round(self.decode_post_ffn_event_time / (self.step-1), 4)) + "ms",
        }    
        self.clear()
        return time_info
    
    def clear(self):
        self.decode_step = 0

        self.prefill_latency = 0
        self.pattern_allocation_event_time_head = 0
        self.prefill_attn_event_time = 0
        self.prefill_write_kv_event_time = 0
        self.prefill_pre_ffn_event_time = 0
        self.prefill_post_ffn_event_time = 0

        self.decode_latency = 0
        self.decode_attn_event_time = 0
        self.decode_write_kv_event_time = 0
        self.prefill_pre_ffn_event_time = 0
        self.decode_pre_ffn_event_time = 0  
        self.decode_post_ffn_event_time = 0

class Full_TimeManager:
    def __init__(self):
        self.decode_step = 0

        self.step = 0
        self.num_layers = 0
        self.num_heads = 0

        self.prefill_latency = 0

        self.decode_latency = 0

        self.prefill_attn_event_time = 0
        self.decode_attn_event_time = 0
        self.forward_attn_start_event = None
        self.forward_attn_end_event = None

        self.prefill_write_kv_event_time = 0
        self.decode_write_kv_event_time = 0
        self.write_kv_start_event = None
        self.write_kv_end_event = None

    def myinit(self, step, num_layers):
        self.decode_step = 0
        self.step = step
        self.num_layers = num_layers
        self.num_heads = num_layers

        self.prefill_latency = 0

        self.decode_latency = 0

        self.prefill_attn_event_time = 0
        self.decode_attn_event_time = 0
        self.forward_attn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]
        self.forward_attn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]

        self.prefill_write_kv_event_time = 0
        self.decode_write_kv_event_time = 0
        self.write_kv_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]
        self.write_kv_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]

        self.prefill_pre_ffn_event_time = 0
        self.decode_pre_ffn_event_time = 0
        self.forward_pre_ffn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]
        self.forward_pre_ffn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]

        self.prefill_post_ffn_event_time = 0
        self.decode_post_ffn_event_time = 0
        self.forward_post_ffn_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]
        self.forward_post_ffn_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.step * self.num_layers)]

    def get_time(self):
        return time.perf_counter()

    def get_final_time(self, prefill_latency, decode_latency):
        torch.cuda.synchronize()
        for layer in range(self.num_layers):
            self.prefill_attn_event_time += self.forward_attn_start_event[layer].elapsed_time(self.forward_attn_end_event[layer]) 
            self.prefill_write_kv_event_time += self.write_kv_start_event[layer].elapsed_time(self.write_kv_end_event[layer]) 

        for layer in range(self.num_layers):
            self.prefill_pre_ffn_event_time += self.forward_pre_ffn_start_event[layer].elapsed_time(self.forward_pre_ffn_end_event[layer]) 
            self.prefill_post_ffn_event_time += self.forward_post_ffn_start_event[layer].elapsed_time(self.forward_post_ffn_end_event[layer])

        for step in range(1, self.decode_step):
            for layer in range(self.num_layers):
                self.decode_attn_event_time += self.forward_attn_start_event[step * self.num_heads + layer].elapsed_time(self.forward_attn_end_event[step * self.num_heads + layer]) 
                self.decode_write_kv_event_time += self.write_kv_start_event[step * self.num_heads + layer].elapsed_time(self.write_kv_end_event[step * self.num_heads + layer]) 
                self.decode_pre_ffn_event_time += self.forward_pre_ffn_start_event[step * self.num_heads + layer].elapsed_time(self.forward_pre_ffn_end_event[step * self.num_heads + layer])
                self.decode_post_ffn_event_time += self.forward_post_ffn_start_event[step * self.num_heads + layer].elapsed_time(self.forward_post_ffn_end_event[step * self.num_heads + layer])


        time_info = {
            "prefill_latency": str(round(prefill_latency * 1000, 4)) + "ms",
            "prefill_attn_event_time": str(round(self.prefill_attn_event_time, 4)) + "ms",
            "prefill_write_kv_event_time": str(round(self.prefill_write_kv_event_time, 4)) + "ms",
            "prefill_ffn_event_time": str(round((prefill_latency * 1000 - self.prefill_attn_event_time - self.prefill_write_kv_event_time), 4)) + "ms",
            "prefill_pre_ffn_event_time": str(round(self.prefill_pre_ffn_event_time, 4)) + "ms",
            "prefill_post_ffn_event_time": str(round(self.prefill_post_ffn_event_time, 4)) + "ms",


            "decode_latency": str(round(decode_latency * 1000 / (self.step-1), 4)) + "ms",
            "decode_attn_event_time": str(round(self.decode_attn_event_time / (self.step-1), 4)) + "ms",
            "decode_write_kv_event_time": str(round(self.decode_write_kv_event_time / (self.step-1), 4)) + "ms",
            "decode_ffn_event_time": str(round((decode_latency * 1000 / (self.step-1) - self.decode_attn_event_time / (self.step-1) - self.decode_write_kv_event_time / (self.step-1)), 4)) + "ms",
            "decode_pre_ffn_event_time": str(round(self.decode_pre_ffn_event_time / (self.step-1), 4)) + "ms",
            "decode_post_ffn_event_time": str(round(self.decode_post_ffn_event_time / (self.step-1), 4)) + "ms",
        }    
        self.clear()
        return time_info
    
    def clear(self):
        self.decode_step = 0

        self.prefill_latency = 0
        self.prefill_attn_event_time = 0
        self.prefill_write_kv_event_time = 0
        self.prefill_pre_ffn_event_time = 0
        self.prefill_post_ffn_event_time = 0

        self.decode_latency = 0
        self.decode_attn_event_time = 0
        self.decode_write_kv_event_time = 0
        self.decode_pre_ffn_event_time = 0
        self.decode_post_ffn_event_time = 0

time_manager = TimeManager()
full_time_manager = Full_TimeManager()