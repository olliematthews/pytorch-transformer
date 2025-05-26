from transformer.model.attention import MultiHeadAttention, AttentionHead
import torch as t
import torch.nn as nn


def test_attention_head():
    batch_size = 5
    hidden_size = 64
    n_tokens = 128
    embedding_size = 32

    head = AttentionHead(hidden_size, embedding_size)
    input_ = t.ones((batch_size, n_tokens, hidden_size), dtype=t.float32)
    head(input_)

    ref_head = nn.MultiheadAttention(hidden_size, 1, bias=False)
    print(ref_head.parameters)


def test_multihead():
    batch_size = 5
    hidden_size = 64
    n_tokens = 128
    embedding_size = 32

    attention = MultiHeadAttention(hidden_size, embedding_size, 5)
    attention_ref = nn.MultiheadAttention()
    input_ = t.ones((batch_size, n_tokens, hidden_size), dtype=t.float32)
    head(input_)


if __name__ == "__main__":
    test_attention_head()
