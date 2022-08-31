import torch
from torch import nn


class MeanPooling(nn.Module):
    def __init__(
            self,
            dim: int = 1
    ):
        super(MeanPooling, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class MaxPooling(nn.Module):
    def __init__(
            self,
            dim: int = 1,
    ):
        super(MaxPooling, self).__init__()
        self.dim = dim

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
