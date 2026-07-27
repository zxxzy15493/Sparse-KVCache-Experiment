# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
# Add project root to path for vq_method imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import json
import logging
import requests
import torch
from typing import Dict, List, Optional

from vq_method.llama31_patch import VQLlama31ForCausalLM
from vq_method.qwen25_patch import VQQwen2ForCausalLM
from vq_method.retrieval_based.pq_search import initialize_objects, del_objects

TOPP_SAVE_TOPK = os.environ.get("TOPP_SAVE_TOPK")
TOPK_SAVE_TOPP = os.environ.get("TOPK_SAVE_TOPP")

def build_chat(tokenizer, prompt, model_name):
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                add_generation_prompt=True, tokenize=False)
        return prompt

class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        print("1", flush=True)
        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            # model_kwargs = {"attn_implementation": "flash_attention_2"}
            print("2", flush=True)
            model_kwargs = {"attn_implementation": "eager"}
        
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=name_or_path,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                device_map="auto",
                # torch_dtype=torch.bfloat16,
                torch_dtype=torch.float16,
                model_kwargs=model_kwargs,
            )
        except:
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,)
            
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs, )
            assert len(output) == len(prompts)
            # output in the form of a list of list of dictionaries
            # outer list len = batch size
            # inner list len = 1
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            # This is a workaround for the llama3 tokenizer not being able to reproduce the same prompt after tokenization
            # see Issue https://github.com/NVIDIA/RULER/issues/54 for explaination
            if self.pipeline is None:
                tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
                prompt = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results


class PQModel:
    """Product Quantization Cache Model for HuggingFace models (Llama, Qwen).

    This class wraps VQ-compressed models (VQLlama31ForCausalLM, VQQwen2ForCausalLM)
    with support for batch processing.
    """

    def __init__(
        self,
        name_or_path: str,
        model_type: str = "llama",
        fixbudget: bool = True,
        budget: int = 1024,
        compress_ratio: float = 0.1,
        important_ratio: float = 0.5,
        recent_ratio: float = 0.5,
        recent_size: int = 32,
        sink_size: int = 16,
        compressor: str = "pq_search",
        n_subvec_per_head: int = 2,
        n_subbits: int = 6,
        topr: int = 32,
        gqa: bool = True,
        max_seq_len: int = 130000,
        cache_block_size: int = 128,
        global_cache_size: int = 4096,
        cache_topk: int = 32,
        score_func: str = "sum",
        drop_ratio: float = 0,
        max_iter: int = 0,
        preserve_layer: int = 0,
        fixthreshold: float = 0.85,
        **generation_kwargs,
    ) -> None:
        """
        Args:
            name_or_path: Path to model or model identifier
            model_type: Model type, "llama" or "qwen"
            fixbudget: Enable fixed budget mode
            budget: Fixed budget size (used when fixbudget is enabled)
            compress_ratio: KV cache compression ratio
            important_ratio: Ratio of important tokens to retrieve
            recent_ratio: Ratio of recent tokens to preserve
            recent_size: Number of recent tokens to keep
            sink_size: Number of most recent tokens to always keep
            compressor: Compression method, "pq_search" for PQCache
            n_subvec_per_head: Number of PQ subvectors per head
            n_subbits: Bits per PQ subvector
            topr: Top-k tokens to retrieve during decoding
            gqa: Whether to use grouped-query attention
            max_seq_len: Maximum sequence length
            cache_block_size: Block size for cache management
            global_cache_size: Size of global cache
            cache_topk: Number of top-k tokens for cache retrieval
            score_func: Score function ("sum" or "max")
            drop_ratio: Drop ratio for tokens
            max_iter: K-means iterations (0 for auto)
            preserve_layer: Number of layers to preserve without compression
            fixthreshold: Fixed threshold for topp attention
        """
        from transformers import AutoTokenizer, AutoConfig

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model_type = model_type
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop', None)
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens', 64)
        self.compressor = compressor
        # Build config
        config = AutoConfig.from_pretrained(name_or_path)
        config.fixbudget = fixbudget
        config.budget = budget
        config.compress_ratio = compress_ratio
        config.important_ratio = important_ratio
        config.recent_ratio = recent_ratio
        config.recent_size = recent_size
        config.sink_size = sink_size
        config.compressor = compressor
        config.n_subvec_per_head = n_subvec_per_head
        config.n_subbits = n_subbits
        config.topr = topr
        config.gqa = gqa
        config.pp_size = 1
        config.keyformer_mode = False
        config.drop_ratio = drop_ratio
        config.preserve_layer = preserve_layer
        config.score_func = score_func
        config.threshold = 1
        config.fixthreshold = fixthreshold
        config.max_iter = max_iter
        config.device = torch.device("cuda")
        config.mean_v_trick = False

        if compressor == "pq_search":
            config.max_seq_len = max_seq_len
            config.cache_block_size = cache_block_size
            config.global_cache_size = global_cache_size
            config.cache_topk = cache_topk
            initialize_objects(config, model=model_type)

        # Load model based on type
        if model_type == "llama":
            self.model = VQLlama31ForCausalLM.from_pretrained(name_or_path, torch_dtype=torch.bfloat16, config=config)
        elif model_type == "qwen":
            self.model = VQQwen2ForCausalLM.from_pretrained(name_or_path, torch_dtype=torch.bfloat16, config=config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model.patch(config)
        self.model = self.model.eval()

    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        """Process a batch of prompts and return generated outputs.

        Args:
            prompts: List of input prompts

        Returns:
            List of dicts with 'text' key containing generated text
        """
        results = []
        data_id = 0
        for prompt in prompts:
            if TOPK_SAVE_TOPP is not None or TOPP_SAVE_TOPK is not None:
                tmp = TOPK_SAVE_TOPP if TOPK_SAVE_TOPP is not None else TOPP_SAVE_TOPK
                with open(f"record/{tmp}.txt", "a") as f:
                    f.write(f"data:{data_id}\n")
            data_id += 1
            # Tokenize input
            # prompt = build_chat(self.tokenizer, prompt, self.model_type)
            # if self.compressor == "pq_search":
            #     prompt = prompt.rsplit(":", 1)[0]
            inputs = self.tokenizer(
                prompt,
                truncation=False,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

            context_length = inputs.input_ids.shape[-1]
            # print(context_length)
            # Generate (use generation_kwargs temperature, or default to 1.0)
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=self.max_genlen,
                    num_beams=1,
                    do_sample=False,
                    **self.generation_kwargs,
                )[0]

            # Decode output
            generated_text = self.tokenizer.decode(
                output[context_length:],
                skip_special_tokens=True
            )
            # generated_length = output.shape[-1] - context_length
            # print(generated_length)
            # Apply stop tokens if specified
            if self.stop is not None:
                for s in self.stop:
                    generated_text = generated_text.split(s)[0]
            torch.cuda.empty_cache()
            results.append({'text': [generated_text]})

        return results

    def cleanup(self):
        """Clean up PQ objects."""
        del_objects()


class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]
