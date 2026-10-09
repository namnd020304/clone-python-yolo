"""Bước 7: ghép backbone + neck + head thành YOLOv8n."""
import torch
from torch import nn

from .backbone import Backbone
from .head import Head
from .neck import Neck


class YOLOv8n(nn.Module):
    """(B, 3, 640, 640) -> (B, 84, 8400). Bản gốc có khoảng 3,15 triệu tham số."""

    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(list(self.neck(*self.backbone(x))))
