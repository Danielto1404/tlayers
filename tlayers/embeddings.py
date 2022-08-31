from typing import Tuple, Union

import einops
import torch
from torch import nn


class PosEmbedding1d(nn.Module):
    def __init__(
            self,
            dim: int,
            seq: int,
            batch_first: bool = True
    ):
        super(PosEmbedding1d, self).__init__()

        self.dim = dim
        self.seq = seq
        self.batch_first = batch_first

        self.embeddings = nn.Parameter(torch.randn(seq, dim))

    def forward(self, x: torch.Tensor):
        assert x.dim() == 3, \
            f"Expected 3D tensor of shape {'(batch, seq, dim)' if self.batch_first else '(seq, batch, dim)'}"

        seq, batch, dim = x.size()

        if self.batch_first:
            batch, seq = (seq, batch)

        assert dim == self.dim, \
            f"Invalid embedding dimension. Expected: {self.dim}, but got: {dim}"

        assert seq == self.seq, \
            f"Invalid sequence length. Expected: {self.seq}, got: {seq}"

        if self.batch_first:
            x = x + self.embeddings
        else:
            x = x.transpose(0, 1) + self.embeddings
            x = x.transpose(0, 1)

        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim}, seq={self.seq}, batch_first={self.batch_first}) "


class PatchesEmbedding(nn.Module):
    def __init__(
            self,
            in_channels: int,
            embedding_dim: int,
            patch_size: Union[int, Tuple[int, int]] = 16,
    ):
        super(PatchesEmbedding, self).__init__()

        self.embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.embedding(x)
        x = einops.rearrange(x, "b d h w -> b (h w) d")

        return x