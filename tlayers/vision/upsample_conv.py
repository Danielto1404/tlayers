from typing import Union, Optional, Tuple

from torch import nn


class UpsampleConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple[int, int]],
            stride: Union[int, tuple[int, int]] = 1,
            padding: Union[str, int, tuple[int, int]] = 0,
            dilation: Union[int, tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
            upsample_size: Optional[Union[int, Tuple[int, ...]]] = None,
            upsample_scale_factor: Optional[Union[float, tuple[float, ...]]] = None,
            upsample_mode: str = 'nearest',
            align_corners: Optional[bool] = None,
            recompute_scale_factor: Optional[bool] = None,
    ):
        super(UpsampleConv2d, self).__init__()

        self.upsample = nn.Upsample(
            upsample_size,
            upsample_scale_factor,
            upsample_mode,
            align_corners,
            recompute_scale_factor
        )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)

        return x
