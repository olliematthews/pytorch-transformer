import torch as t
import torch.nn as nn
import einops
from jaxtyping import Array, Float


class AttentionHead(nn.Module):
    def __init__(self, input_size: int, embedding_size: int):
        super().__init__()
        self.k = t.empty(input_size, embedding_size)
        self.q = t.empty(input_size, embedding_size)
        self.v = t.empty(input_size, embedding_size)

        t.nn.init.xavier_uniform_(self.k)
        t.nn.init.xavier_uniform_(self.q)
        t.nn.init.xavier_uniform_(self.v)

    def forward(self, input: Float[Array, "b n h"]) -> Float[Array, "b n e"]:
        keys = einops.einsum(
            input,
            self.k,
            "b n h, h e -> b n e",
        )
        queries = einops.einsum(
            input,
            self.q,
            "b n h, h e -> b n e",
        )
        values = einops.einsum(
            input,
            self.v,
            "b n h, h e -> b n e",
        )

        attention_map = einops.einsum(
            keys,
            values,
            "b nk e, b nv e -> b nk nv",
        ) / (self.k.shape[1] ** 0.5)
        # Causal masking
        mask = ~t.triu(t.ones_like(attention_map, dtype=t.bool))
        attention_map[mask] = -float("inf")
        attentions = t.softmax(attention_map, dim=-1)
        return einops.einsum(attentions, values, "b nk nv, b nv e -> b nk e")


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, n_heads: int):
        self.heads = [AttentionHead(input_size, embedding_size) for _ in range(n_heads)]
        self.proj = t.nn.Linear(n_heads * embedding_size, input_size)

    def forward(self, input: Float[Array, "b n h"]) -> Float[Array, "b n h"]:
        head_outputs = t.concat([h(input) for h in self.heads], dim=-1)
        return self.proj(head_outputs)
