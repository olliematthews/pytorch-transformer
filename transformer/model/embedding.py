import torch as t
import torch.nn as nn
import einops
from jaxtyping import Array, Float


class RotationalPositionalEmbedding(nn.Module):
    def __init__(self, feature_size: int, n=10000):
        super().__init__()
        self.feature_size = feature_size
        self.n = n

    def forward(
        self, tokens: Float[Array, "batch_size n_tokens vocab_size"]
    ) -> Float[Array, "batch_size n_tokens feature_size"]:
        n_tokens = tokens.shape[1]
        ret = einops.einsum(
            t.arange(n_tokens),
            t.pow(
                self.n,
                -2
                * (t.arange(self.feature_size // 2))
                / t.tensor(self.feature_size, dtype=t.float32),
            ),
            "n_tokens, d_2 -> n_tokens d_2",
        )
        ret = ret[:, :, None].repeat(1, 1, 2)
        ret[:, :, 0].sin_()
        ret[:, :, 1].cos_()
        embeds = einops.rearrange(ret, "n_tokens d_2 k -> n_tokens (d_2 k)")
        return embeds[None, :, :].repeat(tokens.shape[0], 1, 1)
