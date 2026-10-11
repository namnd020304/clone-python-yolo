"""Bước 13: loss của YOLOv8 = 7.5 * box (CIoU) + 0.5 * cls (BCE) + 1.5 * dfl."""
import torch
import torch.nn.functional as F

from .assigner import TaskAlignedAssigner
from .postprocess import bbox_ciou


def dfl_loss(pred_dist: torch.Tensor, target: torch.Tensor, reg_max: int = 16) -> torch.Tensor:
    """Distribution Focal Loss. pred_dist (N, 4, reg_max) logit, target (N, 4) khoảng cách thật theo đơn vị ô.

    Khoảng cách thật (ví dụ 3.3) nằm giữa hai bin nguyên 3 và 4. Loss đẩy xác suất về hai bin đó, trọng số theo độ
    gần: bin 3 nhận 0.7, bin 4 nhận 0.3. Kỳ vọng của phân bố khi đó đúng bằng 3.3.
    Trả về (N,): trung bình trên 4 cạnh.
    """
    target = target.clamp(0, reg_max - 1 - 0.01)
    tl = target.long()          # bin bên trái
    tr = tl + 1                 # bin bên phải
    wl = tr - target            # trọng số bin trái
    wr = 1 - wl
    logp = F.log_softmax(pred_dist, dim=-1)
    loss = -(logp.gather(-1, tl[..., None]).squeeze(-1) * wl + logp.gather(-1, tr[..., None]).squeeze(-1) * wr)
    return loss.mean(-1)


class DetectionLoss:
    """Nhận đầu ra thô của Head ở chế độ train và nhãn của batch, trả về loss (3,) = [box, cls, dfl] đã nhân hệ số.

    Các bước:
      1. Decode DFL ra box dự đoán (xyxy, đơn vị ô).
      2. TAL gán mỗi ô một gt (hoặc nền), cho nhãn class mềm target_scores.
      3. cls: BCE giữa logit class và target_scores trên mọi ô (ô nền học về 0).
      4. box: 1 - CIoU, chỉ trên ô dương, trọng số theo target_scores.
      5. dfl: DFL loss của khoảng cách tới 4 cạnh gt, chỉ trên ô dương, cùng trọng số.
    Cả ba chia cho tổng target_scores (xấp xỉ số vật trong batch).
    """

    def __init__(self, num_classes: int = 80, reg_max: int = 16, box: float = 7.5, cls: float = 0.5,
                 dfl: float = 1.5):
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.gains = torch.tensor([box, cls, dfl])
        self.assigner = TaskAlignedAssigner(num_classes=num_classes, topk=10, alpha=0.5, beta=6.0)

    def __call__(self, preds, gt_labels: torch.Tensor, gt_bboxes: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        """preds = (box (B, 64, A), cls (B, nc, A), anchors (A, 2), strides (A, 1)) từ Head ở chế độ train.
        gt_labels (B, M, 1), gt_bboxes (B, M, 4) xyxy pixel, mask_gt (B, M, 1)."""
        box, cls, anchors, strides = preds
        b, _, a = box.shape
        pred_dist = box.permute(0, 2, 1).reshape(b, a, 4, self.reg_max)  # (B, A, 4, 16)
        pred_scores = cls.permute(0, 2, 1)                               # (B, A, nc)

        # 1. decode: kỳ vọng của phân bố -> khoảng cách l, t, r, b -> xyxy theo đơn vị ô
        proj = torch.arange(self.reg_max, dtype=pred_dist.dtype, device=pred_dist.device)
        dist = pred_dist.softmax(-1) @ proj                              # (B, A, 4)
        pred_bboxes = torch.cat([anchors - dist[..., :2], anchors + dist[..., 2:]], dim=-1)

        # 2. gán nhãn (làm bằng pixel)
        target_bboxes, target_scores, fg = self.assigner(
            pred_scores.detach().sigmoid(), pred_bboxes.detach() * strides, anchors * strides,
            gt_labels, gt_bboxes, mask_gt)
        target_bboxes = target_bboxes / strides                          # về đơn vị ô
        ts_sum = target_scores.sum().clamp(min=1)

        # 3. cls
        loss_cls = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction="sum") / ts_sum

        # 4, 5. box và dfl trên ô dương
        weight = target_scores[fg].sum(-1)                               # (N,)
        ciou = bbox_ciou(pred_bboxes[fg], target_bboxes[fg])
        loss_box = ((1 - ciou) * weight).sum() / ts_sum
        target_ltrb = torch.cat([anchors - target_bboxes[..., :2], target_bboxes[..., 2:] - anchors], dim=-1)
        loss_dfl = (dfl_loss(pred_dist[fg], target_ltrb[fg], self.reg_max) * weight).sum() / ts_sum

        return torch.stack([loss_box, loss_cls, loss_dfl]) * self.gains.to(box.device)
