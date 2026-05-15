"""TichuTrunk + SkillEmbedding shape, gradient, and trace tests."""

import torch

from tichu_training.bc.model import SkillEmbedding, TichuTrunk


def test_trunk_shapes_match_config():
    in_dim, out_dim = 256, 512
    trunk = TichuTrunk(in_dim=in_dim, hidden=128, depth=2, out_dim=out_dim)
    x = torch.randn(8, in_dim)
    y = trunk(x)
    assert y.shape == (8, out_dim)


def test_trunk_output_is_finite():
    trunk = TichuTrunk(in_dim=64, hidden=32, depth=2, out_dim=32)
    y = trunk(torch.randn(4, 64))
    assert torch.isfinite(y).all()


def test_trunk_grad_flows_end_to_end():
    trunk = TichuTrunk(in_dim=32, hidden=16, depth=1, out_dim=16)
    x = torch.randn(2, 32, requires_grad=True)
    y = trunk(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Trunk params should all have grads.
    for p in trunk.parameters():
        assert p.grad is not None


def test_trunk_parameters_are_registered():
    trunk = TichuTrunk(in_dim=16, hidden=16, depth=2, out_dim=8)
    n = sum(1 for _ in trunk.parameters())
    assert n > 0


def test_skill_embedding_shape_includes_neutral_bucket():
    emb = SkillEmbedding(num_buckets=10, dim=64)
    assert emb.embedding.weight.shape == (11, 64)
    # Neutral row is the last one.
    assert emb.neutral_index == 10


def test_skill_embedding_lookup():
    emb = SkillEmbedding(num_buckets=10, dim=8)
    indices = torch.tensor([0, 5, 9, 10], dtype=torch.long)
    out = emb(indices)
    assert out.shape == (4, 8)
    # The neutral embedding (idx 10) is row 10 of the weight matrix.
    assert torch.allclose(out[3], emb.embedding.weight[10])


def test_trunk_is_jit_traceable():
    trunk = TichuTrunk(in_dim=16, hidden=8, depth=1, out_dim=4)
    traced = torch.jit.trace(trunk, torch.randn(2, 16))
    y = traced(torch.randn(3, 16))
    assert y.shape == (3, 4)
