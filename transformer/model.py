import torch as t
import torch.nn as nn
import einops
from jaxtyping import Array, Float


class GeLU(nn.Module):
    def forward(
        self, input: Float[Array, "batch n_features"]
    ) -> Float[Array, "batch n_features"]:
        return 0.5 * (1 + t.tanh((2 / t.pi) ** 0.5(input + 0.044715 * input**3)))


class AttentionHead(nn.Module):
    def __init__(self, input_size: int, embedding_size: int):
        self.k = t.empty(input_size, embedding_size)
        self.q = t.empty(input_size, embedding_size)
        self.v = t.empty(input_size, embedding_size)

        t.nn.init.xavier_uniform(self.k)
        t.nn.init.xavier_uniform(self.q)
        t.nn.init.xavier_uniform(self.v)

    def forward(self, input: Float[Array, "b n h"]) -> Float[Array, "b n e"]:
        keys = t.einsum(
            input,
            self.k,
            "b n h, h e -> b n e",
        )
        queries = t.einsum(
            input,
            self.q,
            "b n h, h e -> b n e",
        )
        values = t.einsum(
            input,
            self.v,
            "b n h, h e -> b n e",
        )

        attention_map = t.einsum(
            einops.repeat(keys, "b n e -> b n n e"),
            einops.repeat(queries, "b n e -> b n n e"),
            "b n n e, b n n e -> b n n",
        ) / t.sqrt(self.k.shape[1])
        # Causal masking
        mask = ~t.triu(t.ones_like(attention_map, dtype=t.bool))
        attention_map[mask] = -t.float("inf")
        attentions = t.softmax(attention_map, 2)
        return t.einsum(attentions, values, "b n n, b n e -> b n e")


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size: int, n_heads: int):
        embedding_size = input_size // n_heads
        self.heads = [AttentionHead(input_size, embedding_size) for _ in range(n_heads)]
        self.proj = t.nn.Linear(n_heads * embedding_size, input_size)

    def forward(self, input: Float[Array, "b n h"]) -> Float[Array, "b n h"]:
        head_outputs = t.concat([h(input) for h in self.heads], dim=-1)
        return self.proj(head_outputs)


class GPT(nn.Module):
    pass
