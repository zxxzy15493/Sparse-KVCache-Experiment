# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json
from functools import partial
from typing import Any
from .modules.forward import attn_forward, LlamaDecoderLayer_forward, full_attn_forward, Full_LlamaDecoderLayer_forward
from .configs.model2path import MODEL2PATH

def update_config_path(model_name: str = None):
        assert (
            model_name in MODEL2PATH
        ), f"The model {model_name} you specified is not supported. You are welcome to add it and open a PR :)"
        return MODEL2PATH[model_name]

class MInference:
    def __init__(
        self,
        model_name: str = None,
    ):
        self.model_name = model_name

    def __call__(self, model: Any = None):

        config_path = update_config_path(self.model_name)
        with open(config_path, "r") as f:
            best_pattern = json.load(f)

        Attention = model.model.layers[0].self_attn.__class__
        DecoderLayer = model.model.layers[0].__class__

        forward = partial(
            attn_forward,
            best_pattern=best_pattern
        )

        def update_module(m):
            if isinstance(m, Attention):
                m.forward = (
                    lambda self, *args, **kwargs: forward(self, *args, **kwargs)
                ).__get__(m, Attention)

            # if isinstance(m, DecoderLayer):
            #     m.forward = LlamaDecoderLayer_forward.__get__(m, DecoderLayer)

        model.apply(update_module)

        for layer_idx, layer in enumerate(model.model.layers):
            layer.layer_idx = layer_idx
            layer.num_layers = len(model.model.layers)
            layer.forward = LlamaDecoderLayer_forward.__get__(layer, DecoderLayer)
            
        return model

    def full_patch(self, model: Any = None):

        Attention = model.model.layers[0].self_attn.__class__
        DecoderLayer = model.model.layers[0].__class__

        forward = partial(
            full_forward,
        )

        def update_module(m):
            if isinstance(m, Attention):
                m.forward = (
                    lambda self, *args, **kwargs: forward(self, *args, **kwargs)
                ).__get__(m, Attention)

        model.apply(update_module)

        for layer_idx, layer in enumerate(model.model.layers):
            layer.layer_idx = layer_idx
            layer.num_layers = len(model.model.layers)
            layer.forward = Full_LlamaDecoderLayer_forward.__get__(layer, DecoderLayer)
            
        return model

        
