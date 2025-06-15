from transformer.model.embedding import RotationalPositionalEmbedding
import torch as t
import torch.nn as nn


def test_rotational_embedding():
    vocab_size = 128
    d = 4
    n = 100
    tokens = t.arange(8)[None, :, None].repeat(1, 1, vocab_size)
    print(tokens.shape)
    embedding_layer = RotationalPositionalEmbedding(feature_size=d, n=n)

    ret = embedding_layer(tokens)
    print(ret.shape)
    print(ret[0])
    for i, row in enumerate(ret[0]):
        for j, val in enumerate(row):
            rad = i / t.pow(n, 2 * (j // 2) / t.tensor(d, dtype=t.float32))
            if j % 2:
                comp = t.cos(rad)
            else:
                comp = t.sin(rad)
            print(comp, val)
            assert t.isclose(val, comp).item()
