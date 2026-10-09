"""Bước 4: backbone của YOLOv8n."""
import torch
from torch import nn

from .blocks import C2f, SPPF, Conv


class Backbone(nn.Module):
    """Thứ tự lớp (kênh đã nhân width 0.25, số lặp đã nhân depth 0.33):

        0 Conv(3, 16, 3, 2)      -> 320x320
        1 Conv(16, 32, 3, 2)     -> 160x160
        2 C2f(32, 32, n=1, shortcut=True)
        3 Conv(32, 64, 3, 2)     -> 80x80
        4 C2f(64, 64, n=2, shortcut=True)     -> P3
        5 Conv(64, 128, 3, 2)    -> 40x40
        6 C2f(128, 128, n=2, shortcut=True)   -> P4
        7 Conv(128, 256, 3, 2)   -> 20x20
        8 C2f(256, 256, n=1, shortcut=True)
        9 SPPF(256, 256)                      -> P5

    forward trả về (p3, p4, p5).
    """

    def __init__(self, c_in: int = 3, c_out: int = 256):
        super().__init__()
        self.conv = Conv(c_in, 16, k=3, s=2)
        self.conv2 = Conv(16, 32, k=3, s=2)
        self.c2f1 = C2f(32, 32, n=1, shortcut=True)
        self.conv3 = Conv(32, 64, k=3, s=2)
        self.c2f2 = C2f(64, 64, n=2, shortcut=True)
        self.conv4 = Conv(64, 128, k=3, s=2)
        self.c2f3 = C2f(128, 128, n=2, shortcut=True)
        self.conv5 = Conv(128, 256, k=3, s=2)
        self.c2f4 = C2f(256, 256, n=1, shortcut=True)
        self.sppf = SPPF(256, 256)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x= self.conv2(x)
        x = self.c2f1(x)
        x = self.conv3(x)
        p3 = self.c2f2(x)
        x = self.conv4(p3)
        p4 = self.c2f3(x)
        x = self.conv5(p4)
        p5 = self.c2f4(x)
        p5 = self.sppf(p5)
        return p3, p4, p5
