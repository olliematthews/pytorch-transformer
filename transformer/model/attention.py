import torch as t
import torch.nn as nn
import einops
from jaxtyping import Array, Float


class AttentionHead(nn.Module):
    def __init__(self, input_size: int, embedding_size: int):
        super().__init__()
        self.w = nn.Linear(input_size, embedding_size * 3, bias=False)
        self.input_size = input_size
        self.embedding_size = embedding_size

    def forward(self, input: Float[Array, "b n h"]) -> Float[Array, "b n e"]:
        qkv = self.w(input)

        q, k, v = qkv.chunk(3, -1)
        attention_map = einops.einsum(
            k,
            q,
            "b nk e, b nv e -> b nk nv",
        ) / (self.embedding_size**0.5)
        # Causal masking
        mask = ~t.triu(t.ones_like(attention_map, dtype=t.bool))
        attention_map[mask] = -float("inf")
        attentions = t.softmax(attention_map, dim=-1)
        return einops.einsum(attentions, v, "b nk nv, b nv e -> b nk e")


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.input_size = input_size
        self.embedding_size = input_size // n_heads

        self.w = nn.Linear(input_size, self.input_size * 3, bias=False)
        self.proj = t.nn.Linear(n_heads * self.embedding_size, input_size, bias=False)

    def forward(self, input: Float[Array, "b n h"]) -> Float[Array, "b n h"]:
        qkv = self.w(input)

        q, k, v = einops.rearrange(
            qkv,
            "b n (triple heads e) -> triple b heads n e",
            heads=self.n_heads,
            triple=3,
        )

        attention_map = einops.einsum(
            k,
            q,
            "b heads nk e, b heads nv e -> b heads nk nv",
        ) / (self.embedding_size**0.5)
        # Causal masking
        mask = ~t.triu(t.ones_like(attention_map, dtype=t.bool))
        attention_map[mask] = -float("inf")
        attentions = t.softmax(attention_map, dim=-1)
        outputs = einops.einsum(
            attentions, v, "b heads nk nv, b heads nv e -> b heads nk e"
        )
        concatted = einops.rearrange(outputs, "b heads nk e -> b nk (heads e)")
        return self.proj(concatted)


# class MultiHeadAttention(nn.Module):
#     def __init__(self, input_size: int, n_heads: int):
#         super().__init__()
#         self.n_heads = n_heads
#         self.input_size = input_size
#         self.embedding_size = input_size // n_heads
#         self.heads = [
#             AttentionHead(input_size, self.embedding_size) for _ in range(n_heads)
#         ]
#         self.proj = t.nn.Linear(n_heads * self.embedding_size, input_size, bias=False)

#     def forward(self, input: Float[Array, "b n h"]) -> Float[Array, "b n h"]:
#         head_outputs = t.concat([h(input) for h in self.heads], dim=-1)
#         return self.proj(head_outputs)
