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

import json
import logging
import requests
import torch
from typing import Dict, List, Optional


def _find_subsequence_positions(sequence: List[int], pattern: List[int]) -> List[int]:
    if not pattern or len(pattern) > len(sequence):
        return []

    return [
        idx for idx in range(len(sequence) - len(pattern) + 1)
        if sequence[idx: idx + len(pattern)] == pattern
    ]

class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, 
            trust_remote_code=True, 
            )

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        
        # try:
        #     self.pipeline = pipeline(
        #         "text-generation",
        #         model=name_or_path,
        #         tokenizer=self.tokenizer,
        #         trust_remote_code=True,
        #         device_map="auto",
        #         torch_dtype=torch.bfloat16,
        #         model_kwargs=model_kwargs,
        #     )
        # except:
        #     self.pipeline = None
        #     self.model = AutoModelForCausalLM.from_pretrained(name_or_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,)
        
        self.pipeline = None
        self.model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            trust_remote_code=True, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            )

        self.model.config.pad_token_id = self.tokenizer.eos_token_id
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

            #print(type(prompts))
            #prompts = [prompts[0] ]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
            #print("input_ids shape:", inputs["input_ids"].shape)
            #print(inputs["attention_mask"].sum(dim=1).tolist())
            #print(" batch  token :", inputs["attention_mask"].sum().item())
            #print("tokenizer.model_max_length =", self.tokenizer.model_max_length)
            #print("model max_position_embeddings =", getattr(self.model.config, "max_position_embeddings", None))
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

    def get_token_ids(self, text: str) -> Dict[str, object]:
        tokenized = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = tokenized.input_ids[0].tolist()
        return {
            "text": text,
            "token_ids": token_ids,
            "first_token_id": token_ids[0] if token_ids else None,
        }

    def get_magic_number_token_ids(self, prompt: str, magic_number: str) -> Dict[str, object]:
        magic_number = str(magic_number)
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0].tolist()

        magic_ids = self.tokenizer(
            magic_number,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0].tolist()

        # Numeric answers in prompts are often preceded by a space.
        magic_ids_with_space = self.tokenizer(
            f" {magic_number}",
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0].tolist()

        positions_no_space = _find_subsequence_positions(prompt_ids, magic_ids)
        positions_with_space = _find_subsequence_positions(prompt_ids, magic_ids_with_space)

        return {
            "magic_number_text": magic_number,
            "magic_number_token_ids": magic_ids,
            "magic_number_token_ids_with_space": magic_ids_with_space,
            "positions_in_prompt": {
                "without_space": positions_no_space,
                "with_space": positions_with_space,
            },
            "first_match_token_index": (
                positions_with_space[0]
                if positions_with_space
                else (positions_no_space[0] if positions_no_space else None)
            ),
        }
    


    # def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
    #     max_gen=100
    #     for prompt in prompts:    # prompt 
    #         #promptsinput  context  question 
    #         if "<QUESTION_START>" in prompt:       # question  context
    #             start = prompt.index("<QUESTION_START>")

    #             # question “question ”
    #             #question = prompt[start:end + len("<QUESTION_END>")]
    #             question = prompt[start + len("<QUESTION_START>"):]

    #             # context  =  question 
    #             prompt = prompt[:start]                               #model.device
    #         input = self.tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")   #context
    #         q_input = self.tokenizer(question, truncation=False, return_tensors="pt").to("cuda") #
    #         q_input.input_ids = q_input.input_ids[:, 1:]    
    #         context_length = input.input_ids.shape[-1] + q_input.input_ids.shape[-1]

    #         #pipeline   
    #         if self.pipeline is None:
    #             with torch.no_grad():   #    # cache   
    #                 output = self.model(input_ids=input.input_ids,past_key_values=None,use_cache=True,)
    #                 past_key_values = output.past_key_values
    #                 for input_id in q_input.input_ids[0]:   #token past_key_values
    #                     output = self.model(
    #                         input_ids=input_id.unsqueeze(0).unsqueeze(0),
    #                         past_key_values=past_key_values,   
    #                         use_cache=True,)
    #                     past_key_values = output.past_key_values
    #                 pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1) #logittoken
    #                 generated_content = [pred_token_idx.item()]
    #                 for _ in range(max_gen - 1):     #greedy   token 
    #                     outputs = self.model(
    #                         input_ids=pred_token_idx,    #logitspastkv
    #                         past_key_values=past_key_values,
    #                         use_cache=True,
    #                     )
    #                     past_key_values = outputs.past_key_values
    #                     pred_token_idx = (
    #                         outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    #                     )
    #                     generated_content += [pred_token_idx.item()]   #pythonlist token id
    #                     if pred_token_idx.item() == self.tokenizer.eos_token_id:  # decode 
    #                         break
    #         #
    #     pred = self.tokenizer.decode(generated_content, skip_special_tokens=True)

    #     # pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    #         # inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
    #         # generated_ids = self.model.generate(
    #         #     **inputs,
    #         #     **self.generation_kwargs
    #         # )
    #         # generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    #     #
    #     results=[]     #
    #     # for text, prompt in zip(pred, prompts):
    #     #     # remove the input form the generated text
    #     #     # This is a workaround for the llama3 tokenizer not being able to reproduce the same prompt after tokenization
    #     #     # see Issue https://github.com/NVIDIA/RULER/issues/54 for explaination
    #     #     if self.pipeline is None:
    #     #         tokenized_prompt = self.tokenizer(prompt, return_tensors="pt", padding=True)
    #     #         prompt = self.tokenizer.decode(tokenized_prompt.input_ids[0], skip_special_tokens=True)
    #     #     if text.startswith(prompt):
    #     #         text = text[len(prompt):]

    #     #     if self.stop is not None:
    #     #         for s in self.stop:
    #     #             text = text.split(s)[0]

    #     results.append({'text': [pred]})

    #     return results


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

    def get_token_ids(self, text: str) -> Dict[str, object]:
        tokenized = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = tokenized.input_ids[0].tolist()
        return {
            "text": text,
            "token_ids": token_ids,
            "first_token_id": token_ids[0] if token_ids else None,
        }

    def get_magic_number_token_ids(self, prompt: str, magic_number: str) -> Dict[str, object]:
        magic_number = str(magic_number)
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0].tolist()

        magic_ids = self.tokenizer(
            magic_number,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0].tolist()

        magic_ids_with_space = self.tokenizer(
            f" {magic_number}",
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0].tolist()

        positions_no_space = _find_subsequence_positions(prompt_ids, magic_ids)
        positions_with_space = _find_subsequence_positions(prompt_ids, magic_ids_with_space)

        return {
            "magic_number_text": magic_number,
            "magic_number_token_ids": magic_ids,
            "magic_number_token_ids_with_space": magic_ids_with_space,
            "positions_in_prompt": {
                "without_space": positions_no_space,
                "with_space": positions_with_space,
            },
            "first_match_token_index": (
                positions_with_space[0]
                if positions_with_space
                else (positions_no_space[0] if positions_no_space else None)
            ),
        }
