import torch as t
import torch.nn as nn
import torch as t
import torch.nn as nn
import einops
from jaxtyping import Array, Float


class GeLU(nn.Module):
    def forward(
        self, input: Float[Array, "batch n_tokens n_features"]
    ) -> Float[Array, "batch n_tokens n_features"]:
        return 0.5 * (1 + t.tanh((2 / t.pi) ** 0.5(input + 0.044715 * input**3)))


class MLP(t.Module):
    def __init__(
        self, residual_size: int, hidden_dimension_size: int, bias: bool = False
    ):
        super().__init__()
        self.proj_up = nn.Linear(residual_size, hidden_dimension_size, bias)
        self.gelu = GeLU(hidden_dimension_size)
        self.proj_down = nn.Linear(residual_size, hidden_dimension_size, bias)

    def __call__(
        self, input_: Float[Array, "batch n_tokens n_features"]
    ) -> Float[Array, "batch n_tokens n_features"]:
        return self.proj_down(self.gelu(self.proj_up(input_)))
