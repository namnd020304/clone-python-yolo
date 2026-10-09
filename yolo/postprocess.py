"""Bước 9: từ đầu ra (1, 84, 8400) lấy ra danh sách box cuối cùng."""
import torch


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU giữa từng cặp box xyxy: a (N, 4), b (M, 4) -> (N, M). Viết bằng tensor, không dùng vòng for."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=2)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-9)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    """Non-maximum suppression: sắp xếp theo điểm giảm dần, giữ box điểm cao nhất, bỏ các box còn lại có
    IoU > iou_thres với nó, lặp lại. Trả về chỉ số (index) các box được giữ.

    Không dùng torchvision.ops.nms; chỉ dùng nó trong check.py để so kết quả.
    """
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        order = rest[box_iou(boxes[i][None], boxes[rest])[0] <= iou_thres]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long)


def postprocess(pred: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.45) -> torch.Tensor:
    """pred (1, 84, 8400), box dạng (cx, cy, w, h).

    1. Đổi về (8400, 84); điểm của mỗi ô = max trên 80 class, kèm class id.
    2. Bỏ ô có điểm < conf_thres.
    3. Đổi box sang xyxy, chạy NMS theo từng class (mẹo: cộng class_id * 4096 vào toạ độ trước khi NMS).
    Trả về (K, 6): [x1, y1, x2, y2, score, class_id].
    """
    p = pred[0].T  # (8400, 84)
    scores, cls = p[:, 4:].max(dim=1)
    mask = scores >= conf_thres
    p, scores, cls = p[mask], scores[mask], cls[mask]
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    keep = nms(boxes + cls[:, None].float() * 4096, scores, iou_thres)
    return torch.cat([boxes[keep], scores[keep, None], cls[keep, None].float()], dim=1)
