"""Lightweight SparQ model-conversion interface."""

from __future__ import annotations

from typing import Any

from transformers import PreTrainedModel

from .methods import ann_attention_copy


class Sparsity:
  name: str

  def __init__(self, name: str, **kwargs: Any):
    self.name = name
    self.__dict__.update(kwargs)

  def __str__(self) -> str:
    return str(self.__dict__)


class SparsityMethods:
  @classmethod
  def apply(cls, sparsity: Sparsity, model: PreTrainedModel) -> PreTrainedModel:
    method = getattr(cls, sparsity.name)
    return method(
      model, **{k: v for k, v in sparsity.__dict__.items() if k != "name"}
    )

  @staticmethod
  def dense(model: PreTrainedModel) -> PreTrainedModel:
    return model

  @staticmethod
  def ann(model: PreTrainedModel, **settings: Any) -> PreTrainedModel:
    supported_model = isinstance(model, ann_attention_copy.Model.__args__)
    assert supported_model or ann_attention_copy._looks_like_chatglm_model(model)
    return ann_attention_copy.convert(model, ann_attention_copy.Settings(**settings))
