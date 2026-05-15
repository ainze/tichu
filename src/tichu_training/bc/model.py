"""Shared MLP trunk and skill embedding (per ADR-001).

Architecture: 4 residual blocks of width 1024, GELU + LayerNorm, 512-dim
trunk output. Defaults are documented in the ADR; the test suite uses
smaller widths for speed.
"""

import torch
from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(self.ln1(x)))
        h = self.fc2(self.ln2(h))
        return x + h


class TichuTrunk(nn.Module):
    """Residual MLP trunk producing a `out_dim`-dim representation.

    Inputs are the engineered featurizer outputs (optionally with a skill
    embedding concatenated). Downstream task heads consume the trunk output.
    """

    def __init__(
        self,
        in_dim: int,
        *,
        hidden: int = 1024,
        depth: int = 4,
        out_dim: int = 512,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([_ResidualBlock(hidden) for _ in range(depth)])
        self.output_proj = nn.Linear(hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class SkillEmbedding(nn.Module):
    """Embedding table with `num_buckets + 1` rows.

    The first `num_buckets` rows are the decile buckets; the final row is the
    neutral / cold-start embedding (handle the `skill_decile IS NULL` case in
    parquet by passing `neutral_index` for those rows).
    """

    def __init__(self, num_buckets: int = 10, dim: int = 64) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.neutral_index = num_buckets
        self.embedding = nn.Embedding(num_buckets + 1, dim)

    def forward(self, decile_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(decile_indices)
