"""Bước 6: head tách nhánh (decoupled), không dùng anchor box (anchor-free), có DFL."""
import math

import torch
from torch import nn

from .blocks import Conv


def make_anchors(feats: list[torch.Tensor], strides: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Tâm từng ô (x + 0.5, y + 0.5) theo đơn vị ô của mức đó, và stride của ô.

    Trả về anchors (A, 2) và strides (A, 1), A = tổng h * w của các mức (8400 với ảnh 640).
    """
    anchors, stride_t = [], []
    for feat, s in zip(feats, strides):
        h, w = feat.shape[2:]
        ys, xs = torch.meshgrid(torch.arange(h, device=feat.device), torch.arange(w, device=feat.device),
                                indexing="ij")
        anchors.append(torch.stack([xs, ys], dim=-1).view(-1, 2).float() + 0.5)
        stride_t.append(torch.full((h * w, 1), float(s), device=feat.device))
    return torch.cat(anchors), torch.cat(stride_t)


class DFL(nn.Module):
    """Distribution Focal Loss decode: mỗi cạnh box được dự đoán dưới dạng phân bố trên reg_max = 16 giá trị
    0..15. Lấy softmax rồi tính kỳ vọng: khoảng cách = sum(softmax(p) * [0, 1, ..., 15]).

    Vào (B, 4 * 16, A) -> ra (B, 4, A), A = số ô (8400).
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.arange(reg_max, dtype=torch.float32).view(1, 1, -1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        x = x.view(b, 4, self.reg_max, a).softmax(dim=2)
        return (x * self.project).sum(dim=2)


class Head(nn.Module):
    """Với mỗi mức i (kênh vào 64, 128, 256; stride 8, 16, 32):
        nhánh box : Conv(c_i, 64, 3) -> Conv(64, 64, 3) -> nn.Conv2d(64, 4 * 16, 1)
        nhánh cls : Conv(c_i, 80, 3) -> Conv(80, 80, 3) -> nn.Conv2d(80, num_classes, 1)
    Nối 3 mức -> box (B, 64, 8400), cls (B, num_classes, 8400).

    Decode:
      - DFL trên 64 kênh box -> khoảng cách (trái, trên, phải, dưới) tính theo đơn vị ô.
      - Tâm ô (anchor point) = (x + 0.5, y + 0.5). Box xyxy = tâm -/+ khoảng cách, rồi nhân stride
        để ra pixel. Đổi sang (cx, cy, w, h).
      - sigmoid trên các kênh class.
    Ở chế độ eval, forward trả về (B, 4 + num_classes, 8400) = (B, 84, 8400).
    Ở chế độ train (model.train()), forward trả về đầu ra thô cho loss:
        box (B, 64, 8400) logit DFL, cls (B, num_classes, 8400) logit, anchors (8400, 2), strides (8400, 1).
    """

    def __init__(self, num_classes: int = 80, channels: tuple[int, int, int] = (64, 128, 256),
                 strides: tuple[int, int, int] = (8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        c_box, c_cls = 64, max(channels[0], min(num_classes, 100))
        self.box_branches = nn.ModuleList(
            nn.Sequential(Conv(c, c_box, 3), Conv(c_box, c_box, 3), nn.Conv2d(c_box, 4 * 16, 1)) for c in channels)
        self.cls_branches = nn.ModuleList(
            nn.Sequential(Conv(c, c_cls, 3), Conv(c_cls, c_cls, 3), nn.Conv2d(c_cls, num_classes, 1)) for c in channels)
        self.dfl = DFL()
        self.init_bias()

    def init_bias(self):
        """Khởi tạo bias lớp cuối giống Ultralytics để train từ đầu ổn định: box = 2.0; class sao cho sigmoid ban đầu
        nhỏ (khoảng 5 vật trên ảnh 640, chia đều cho các class và các ô của mức)."""
        for box_b, cls_b, s in zip(self.box_branches, self.cls_branches, self.strides):
            nn.init.constant_(box_b[-1].bias, 2.0)
            nn.init.constant_(cls_b[-1].bias, math.log(5 / self.num_classes / (640 / s) ** 2))

    def forward(self, feats: list[torch.Tensor]):
        box = torch.cat([b(f).flatten(2) for f, b in zip(feats, self.box_branches)], dim=2)  # (B, 64, A)
        cls = torch.cat([c(f).flatten(2) for f, c in zip(feats, self.cls_branches)], dim=2)  # (B, nc, A)
        anchors, strides = make_anchors(feats, self.strides)                              # (A, 2), (A, 1)
        if self.training:
            return box, cls, anchors, strides
        return self.decode(box, cls, anchors, strides)

    def decode(self, box: torch.Tensor, cls: torch.Tensor, anchors: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
        """Đầu ra thô (như ở chế độ train) -> (B, 4 + num_classes, A) như ở chế độ eval."""
        dist = self.dfl(box)        # (B, 4, A): l, t, r, b theo đơn vị ô
        anchors = anchors.T[None]   # (1, 2, A)
        strides = strides.T[None]   # (1, 1, A)
        x1y1 = anchors - dist[:, :2]
        x2y2 = anchors + dist[:, 2:]
        cxcy = (x1y1 + x2y2) / 2 * strides
        wh = (x2y2 - x1y1) * strides
        return torch.cat([cxcy, wh, cls.sigmoid()], dim=1)
