import time
from numpy import rint
import torch
from termcolor import colored


class LLM:
    """
    A class representing the LLM (currently support Llama, Qwen and GLM).
    """
    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str,
        RECALL: bool = False,
        budget: int = 128
    ) -> None:
        """ Initializes the LLM.
        Args:
            model_name (str): The name of the model.
            max_length (int): The maximum length (prefill+decode) of sequences.
            dtype (torch.dtype): The data type for model computations.
            device_map (str): The device for model, suppor 'cuda:x' or 'auto (automatically use all visible GPUs)'.
        """
        self.RECALL = RECALL
        self.budget = budget

        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map 
        self.device = "cuda:0" 

        
        self.prefill_latency = 0
        self.decode_latency = 0
        self.Latency = 0
        self.TPOT = 0

    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        # print(f'Layer = {layer_idx}, start_bdx = {start_bdx}')

        bsz, seq_len, dim = hidden_states.shape 
        layer = self.layers[layer_idx]
        
        # original hidden_states used as residual, clone a new one to process
        temp_hidden_states = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            temp_hidden_states[:, start_idx:end_idx, :] = self.layernorm(temp_hidden_states[:, start_idx:end_idx, :], 
                                                                         layer.input_layernorm_variance_epsilon, 
                                                                         layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        del temp_hidden_states
        torch.cuda.empty_cache()
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)       # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        
        self.key_states_shape = key_states.shape
        self.key_states_length = key_states.shape[1]

        key_states, value_states = self.kv_cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx, start_bdx)

        torch.cuda.empty_cache()
        temp_attn_out = self.prefill_attention(query_states, key_states, value_states, layer_idx)

        self.kv_cache.sync(layer_idx, start_bdx)
    
        del query_states, key_states, value_states
        torch.cuda.empty_cache()
        
        hidden_states += self.wo(temp_attn_out, layer, temp_attn_out.shape[0], seq_len, dim)
        del temp_attn_out
        torch.cuda.empty_cache()

        # post attention
        residual = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            hidden_states[:, start_idx:end_idx, :] = self.layernorm(hidden_states[:, start_idx:end_idx, :], 
                                                                    layer.post_attention_layernorm_variance_epsilon, 
                                                                    layer.post_attention_layernorm_weight)
            hidden_states[:, start_idx:end_idx, :] = self.mlp(hidden_states[:, start_idx:end_idx, :], layer)   
        
        hidden_states += residual

        del residual
        torch.cuda.empty_cache()

        return hidden_states

    def layer_decode(self, layer_idx, hidden_states):
        # print(f'Layer = {layer_idx}')

        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]

        hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)

        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)

        # key_states, value_states only use in full attention calculation, in retroinfer attention, they are not used(None).
        key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx) 
        # attn_out shape: [bs, seq_len, num_head, head_dim]
        attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
        

        hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states

        return hidden_states

    def prefill_forward(self, inputs_ids):
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device)
        # start_bdx mean start batch index

        for start_bdx in range(0, bsz, 1):
            end_bdx = min(bsz, start_bdx + 1)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])  # [1, seq_len, hidden_size]

            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    hidden_states = self.parameter_move(hidden_states, ldx)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(self.layers[0].device)
            else:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
      
        
        last_hidden_states = self.layernorm(last_hidden_states.contiguous(), self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden_states)
        
        return logits
        
    def decode_forward(self, inputs_ids):
        hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
        self.key_states_length += 1
        hidden_states = self.layernorm(hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        
        return logits

    def inference(self, inputs_ids, fixed_output_length):
        outputs_ids = []    # multi iteration, multi request
        output_ids = []     # single iteration, multi request
        
        print("Start prefilling ...")
        torch.cuda.synchronize()
        prefill_start = time.perf_counter()

        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = logits.argmax(dim=-1)
        outputs_ids.append(output_ids)
        self.move()

        torch.cuda.synchronize()

        prefill_end = time.perf_counter()
        self.prefill_latency = round((prefill_end - prefill_start), 2)

        print("Start decoding ...")
        decode_start = time.perf_counter()
        self.max_new_length = self.max_new_length if fixed_output_length == 0 else fixed_output_length
        generate_token = 0
        eos_tokens = getattr(self, "eos_tokens", [])
        if not isinstance(eos_tokens, (list, tuple, set)):
            eos_tokens = [eos_tokens]
        stop_token_ids = {t for t in eos_tokens if t is not None}

        model_name_lower = (self.model_name or "").lower()
        # Backward-compatible fallbacks for common chat/end tokens.
        if "llama" in model_name_lower:
            stop_token_ids.update({128008, 128001, 128009})
        if "qwen" in model_name_lower:
            stop_token_ids.add(151645)
        if "glm" in model_name_lower:
            stop_token_ids.update({151329, 151336, 151338})
        if "deepseek" in model_name_lower:
            # DeepSeek-R1 distills may use either Qwen-style <|im_end|> or a plain EOS.
            stop_token_ids.update({151643, 151645})

        for _ in range(self.max_new_length-1):
            logits = self.decode_forward(inputs_ids=output_ids)
            output_ids = logits.argmax(dim=-1)
            outputs_ids.append(output_ids)
            generate_token += 1
            if fixed_output_length == 0 and stop_token_ids and output_ids[0].item() in stop_token_ids:
                print("===" * 80)
                print(colored(f"Stop token {output_ids[0].item()} generated, stopping decoding.", 'yellow'))
                print("===" * 80)
                break

        torch.cuda.synchronize()
        decode_end = time.perf_counter()
        self.decode_latency = round((decode_end - decode_start), 5)
        self.Latency = self.prefill_latency + self.decode_latency

        self.TPOT = round((decode_end - decode_start)  / (generate_token) * 1000, 5)
        print(colored(
            f"Decoding latency: {round((decode_end - decode_start)  / (generate_token), 5)} s/step, "
            f"Throughput: {round(self.batch_size * (generate_token) / (decode_end - decode_start), 2)} tokens/s\n",
            'green'
        ))

        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()
        
        return outputs_ids

    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config=None, prefill_method="Full_Flash_Attn", fixed_output_length=0):
        """ LLM Inference.
        Args:
            attention_type: str,
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
        """
        
        bs, input_length = inputs_ids.shape
        print(colored(f"\ninput_length is: {input_length}\n", 'cyan'))
        assert input_length + max_new_length <= self.max_length, \
        f"Error: input_length({input_length}) + max_new_length({max_new_length}) exceeds max_length({self.max_length})"

        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        self.attention_type = attention_type
        self.prefill_method = prefill_method
        '''
            llm.tokenizer.padding_side = "left"
            valid_start: record the valid start position for each sample in the batch
        '''


        print(colored(f"prefill_method: {self.prefill_method} and attention_type: {self.attention_type}", 'cyan'))
        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks
        torch.cuda.empty_cache()

        print("Allocate GPU buffers and CPU pin memory ...\n")
        self.init_kv_cache(input_length, valid_start, attn_config)

        outputs = self.inference(inputs_ids, fixed_output_length)


        return outputs
