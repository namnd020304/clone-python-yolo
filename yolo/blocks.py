"""Bước 1–3: các khối cơ bản của YOLOv8."""
import torch
from torch import nn


class Conv(nn.Module):
    """Bước 1: Conv2d (không bias) -> BatchNorm2d -> SiLU.

    padding = k // 2 để stride 1 giữ nguyên H, W; stride 2 thì H, W giảm một nửa.
    Ví dụ: Conv(3, 16, k=3, s=2) biến (1, 3, 640, 640) thành (1, 16, 320, 320).
    """

    def __init__(self, c_in: int, c_out: int, k: int = 1, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)  # giống Ultralytics
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Bước 2: hai Conv 3x3 nối tiếp; nếu shortcut=True thì cộng thêm x vào đầu ra (residual)."""

    def __init__(self, c: int, shortcut: bool = True):
        super().__init__()
        self.conv1 = Conv(c, c, k=3, s=1)
        self.conv2 = Conv(c, c, k=3, s=1)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x)) if self.shortcut else self.conv2(self.conv1(x))


class C2f(nn.Module):
    """Bước 2: Conv 1x1 -> tách đôi kênh -> n Bottleneck nối tiếp trên nửa sau,
    giữ lại đầu ra của từng Bottleneck -> nối (concat) tất cả -> Conv 1x1 ra c_out.

    Với hidden = c_out // 2: concat có (2 + n) * hidden kênh.
    """

    def __init__(self, c_in: int, c_out: int, n: int = 1, shortcut: bool = False):
        super().__init__()
        self.hidden = c_out // 2
        self.conv1 = Conv(c_in, 2 * self.hidden, k=1, s=1)
        self.bottlenecks = nn.ModuleList([Bottleneck(self.hidden, shortcut) for _ in range(n)])
        self.conv2 = Conv((2 + n) * self.hidden, c_out, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        ys = [y[:, :self.hidden], y[:, self.hidden:]]  # tách đôi theo kênh (dim 1)
        for bottleneck in self.bottlenecks:
            ys.append(bottleneck(ys[-1]))  # chạy trên kết quả mới nhất, giữ lại mọi kết quả
        return self.conv2(torch.cat(ys, dim=1))


class SPPF(nn.Module):
    """Bước 3: Conv 1x1 (giảm còn c_in // 2 kênh) -> MaxPool2d(5, stride 1, padding 2) ba lần liên tiếp
    -> concat [x, y1, y2, y3] -> Conv 1x1 ra c_out."""

    def __init__(self, c_in: int, c_out: int, k: int = 5):
        super().__init__()
        self.conv1 = Conv(c_in, c_in // 2, k=1, s=1)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.conv2 = Conv(c_in * 2, c_out, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y1 = self.maxpool(y)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.conv2(torch.cat([y, y1, y2, y3], dim=1))
