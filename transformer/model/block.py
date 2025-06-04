from model.attention import MultiHeadAttention
from model.mlp import MLP
import torch as t
import torch.nn as nn
import torch as t
import torch.nn as nn
import einops
from jaxtyping import Array, Float
from dataclasses import dataclass

@dataclass
class BlockConfig:
    residual_size: int
    mha_n_heads: int
    mlp_hidden_dimension: int
    mlp_bias: bool = False

class Block(t.Module):
    def __init__(self, config: BlockConfig):
        self.mlp = MLP(config.residual_size, config.mlp_hidden_dimension, config.mlp_bias)
        self.mha = MultiHeadAttention(config.residual_size, config.mha_n_heads)

    def __call__(self, input_: Float[Array, "batch n_tokens n_features"]) -> Float[Array, "batch n_tokens n_features"]:
        