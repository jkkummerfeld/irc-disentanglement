from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
import torch


@dataclass
class Option:
    text_rep: list  # list[int] for GloVe word IDs
    features: list  # list[float], length == FEATURES
    text_ascii: str


@dataclass
class ForwardMetaData:
    query_idx: int
    filename: str = "unknown"


@dataclass
class ForwardArgs:
    query: list       # list[int] of word IDs for query message
    options: list     # list[Option]
    gold: list        # list[int] of relative indices that are correct
    metadata: ForwardMetaData = None
    is_train: bool = True


@dataclass
class Prediction:
    pred_idx: torch.Tensor   # scalar index of predicted link
    gold_idx: torch.Tensor   # gold link indices
    pred_prob: torch.Tensor  # probability of predicted link
    gold_prob: torch.Tensor  # probability of gold link(s)


class EmbedType(Enum):
    WORD = 1
    SENTENCE = 2


@dataclass
class EmbedInfo:
    name: str
    embed_type: EmbedType
    dim: int = 0


class Embedder(ABC):

    @abstractmethod
    def embed(self, words: list) -> torch.Tensor:
        pass

    @abstractmethod
    def get_embed_info(self) -> EmbedInfo:
        pass
