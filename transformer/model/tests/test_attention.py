from transformer.model.attention import MultiHeadAttention, AttentionHead
import torch as t
import torch.nn as nn


def test_attention_head():
    batch_size = 5
    hidden_size = 64
    n_tokens = 128
    embedding_size = 64

    head = AttentionHead(hidden_size, embedding_size)
    input_ = t.ones((batch_size, n_tokens, hidden_size), dtype=t.float32)
    head(input_)

    ref_head = nn.MultiheadAttention(hidden_size, 1, bias=False)
    with t.no_grad():
        ref_head.in_proj_weight.copy_(head.w.weight)
        t.eye(
            ref_head.out_proj.weight.data.shape[0],
            out=ref_head.out_proj.weight,
        )

    head.eval()
    ref_head.eval()

    res = head(input_)
    ref_res = ref_head(input_, input_, input_, need_weights=False)[0]
    assert t.allclose(res, ref_res, atol=1e-5, rtol=1e-4)


def test_multihead():
    batch_size = 5
    hidden_size = 64
    n_tokens = 128
    # embedding_size = 32
    n_heads = 4

    attention = MultiHeadAttention(hidden_size, n_heads)
    ref_attention = nn.MultiheadAttention(hidden_size, n_heads, bias=False)

    input_ = t.ones((batch_size, n_tokens, hidden_size), dtype=t.float32)
    attention(input_)

    with t.no_grad():
        ref_attention.in_proj_weight.copy_(attention.w.weight)
        ref_attention.out_proj.weight.copy_(attention.proj.weight)

    attention.eval()
    ref_attention.eval()

    res = attention(input_)
    ref_res = ref_attention(input_, input_, input_, need_weights=False)[0]
    assert t.allclose(res, ref_res, atol=1e-5, rtol=1e-4)
