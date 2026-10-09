"""Bước 5: neck PAN-FPN của YOLOv8n."""
import torch
from torch import nn

from .blocks import C2f, Conv


class Neck(nn.Module):
    """Nhận (p3, p4, p5) từ backbone, trả về (n3, n4, n5) cùng shape.

    Đi lên (top-down, upsample x2 bằng nn.Upsample(scale_factor=2, mode="nearest")):
        up(p5) concat p4 -> C2f(384, 128, n=1)            -> t4  (128, 40, 40)
        up(t4) concat p3 -> C2f(192, 64, n=1)             -> n3  (64, 80, 80)
    Đi xuống (bottom-up, Conv 3x3 stride 2):
        Conv(64, 64, 3, 2)(n3) concat t4 -> C2f(192, 128, n=1)    -> n4  (128, 40, 40)
        Conv(128, 128, 3, 2)(n4) concat p5 -> C2f(384, 256, n=1)  -> n5  (256, 20, 20)

    Các C2f trong neck dùng shortcut=False. Concat theo chiều kênh (dim=1).
    """

    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_t4 = C2f(384, 128, n=1)
        self.c2f_n3 = C2f(192, 64, n=1)
        self.down3 = Conv(64, 64, k=3, s=2)
        self.c2f_n4 = C2f(192, 128, n=1)
        self.down4 = Conv(128, 128, k=3, s=2)
        self.c2f_n5 = C2f(384, 256, n=1)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        t4 = self.c2f_t4(torch.cat([self.up(p5), p4], dim=1))
        n3 = self.c2f_n3(torch.cat([self.up(t4), p3], dim=1))
        n4 = self.c2f_n4(torch.cat([self.down3(n3), t4], dim=1))
        n5 = self.c2f_n5(torch.cat([self.down4(n4), p5], dim=1))
        return n3, n4, n5
