from torch import nn
from transformers import AutoModelForCausalLM


class ChatGLMForConditionalGeneration(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.config = model.config
        if hasattr(model, "transformer"):
            self.model = model.transformer
        elif hasattr(model, "model"):
            self.model = model.model
        else:
            self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(model)

    def forward(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)

    def eval(self):
        self.inner_model.eval()
        return super().eval()