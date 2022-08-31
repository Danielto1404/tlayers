import torch
from torch import nn


class _DimPooling(nn.Module):
    def __init__(
            self,
            dim: int = 1
    ):
        super(_DimPooling, self).__init__()
        self.dim = dim

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim})"


class MeanPooling(_DimPooling):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class MaxPooling(_DimPooling):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=self.dim)


class ClsPooling(nn.Module):
    def __init__(
            self,
            cls_position: int = 0,
            batch_first: bool = True
    ):
        super(ClsPooling, self).__init__()
        self.cls_position = cls_position
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            return x[:, self.cls_position]
        else:
            return x[self.cls_position]

    def __repr__(self):
        return f"{self.__class__.__name__}(cls_position={self.cls_position}, batch_first={self.batch_first})"
