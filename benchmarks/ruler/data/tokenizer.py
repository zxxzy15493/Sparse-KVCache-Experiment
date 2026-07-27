# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import List


def select_tokenizer(tokenizer_type: str, tokenizer_path: str):
    if tokenizer_type != "hf":
        raise ValueError("RULER supports only tokenizer_type=hf")
    return HFTokenizer(model_path=tokenizer_path)


class HFTokenizer:
    """Tokenizer used to construct all local RULER datasets."""

    def __init__(self, model_path: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def text_to_tokens(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def tokens_to_text(self, tokens: List[int]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)
