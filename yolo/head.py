"""Bước 6: head tách nhánh (decoupled), không dùng anchor box (anchor-free), có DFL."""
import torch
from torch import nn

from .blocks import Conv


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
    forward trả về (B, 4 + num_classes, 8400) = (B, 84, 8400).
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

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        boxes, clss, anchors, stride_t = [], [], [], []
        for feat, box_b, cls_b, s in zip(feats, self.box_branches, self.cls_branches, self.strides):
            h, w = feat.shape[2:]
            boxes.append(box_b(feat).flatten(2))
            clss.append(cls_b(feat).flatten(2))
            ys, xs = torch.meshgrid(torch.arange(h, device=feat.device), torch.arange(w, device=feat.device),
                                    indexing="ij")
            anchors.append(torch.stack([xs, ys], dim=0).flatten(1).float() + 0.5)  # (2, h*w)
            stride_t.append(torch.full((1, h * w), float(s), device=feat.device))
        dist = self.dfl(torch.cat(boxes, dim=2))          # (B, 4, A): l, t, r, b
        anchors = torch.cat(anchors, dim=1).unsqueeze(0)  # (1, 2, A)
        strides = torch.cat(stride_t, dim=1).unsqueeze(0)  # (1, 1, A)
        x1y1 = anchors - dist[:, :2]
        x2y2 = anchors + dist[:, 2:]
        cxcy = (x1y1 + x2y2) / 2 * strides
        wh = (x2y2 - x1y1) * strides
        return torch.cat([cxcy, wh, torch.cat(clss, dim=2).sigmoid()], dim=1)
