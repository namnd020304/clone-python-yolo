"""Bước 12: Task-Aligned Assigner (TAL) — chọn ô nào chịu trách nhiệm học box thật nào."""
import torch
import torch.nn.functional as F

from .postprocess import bbox_ciou


class TaskAlignedAssigner:
    """Với mỗi box thật (gt), chấm điểm mọi ô bằng

        align = score^alpha * CIoU^beta

    score = điểm model đang dự đoán cho đúng class của gt tại ô đó, CIoU = độ khớp giữa box model dự đoán tại ô đó
    và gt. Ô "tốt" là ô vừa đoán đúng class vừa đoán đúng vị trí, nên nhiệm vụ phân loại và định vị được học cùng
    nhau (task-aligned).

    Các bước:
      1. Chỉ xét ô có tâm nằm trong gt (gt có cạnh < 16 px được nới ra 16 px quanh tâm, giống Ultralytics 8.4).
      2. Mỗi gt lấy top-k ô có align cao nhất làm ô dương.
      3. Một ô được nhiều gt chọn -> giữ gt có CIoU cao nhất.
      4. Nhãn class mềm: one-hot của class gt, nhân với align đã chuẩn hoá sao cho ô tốt nhất của mỗi gt có giá trị
         bằng CIoU lớn nhất của gt đó.

    Mọi toạ độ tính bằng pixel của ảnh 640.
    """

    def __init__(self, num_classes: int = 80, topk: int = 10, alpha: float = 0.5, beta: float = 6.0,
                 min_side: float = 16.0, eps: float = 1e-9):
        self.num_classes = num_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.min_side = min_side
        self.eps = eps

    @torch.no_grad()
    def __call__(self, pd_scores: torch.Tensor, pd_bboxes: torch.Tensor, anchors: torch.Tensor,
                 gt_labels: torch.Tensor, gt_bboxes: torch.Tensor, mask_gt: torch.Tensor):
        """
        pd_scores (B, A, nc) đã qua sigmoid · pd_bboxes (B, A, 4) xyxy · anchors (A, 2) tâm ô
        gt_labels (B, M, 1) · gt_bboxes (B, M, 4) xyxy · mask_gt (B, M, 1): 1 = gt thật, 0 = phần đệm cho đủ M

        Trả về target_bboxes (B, A, 4), target_scores (B, A, nc), fg_mask (B, A) bool.
        """
        b, a, nc = pd_scores.shape
        m = gt_bboxes.shape[1]
        if m == 0:
            return torch.zeros_like(pd_bboxes), torch.zeros_like(pd_scores), pd_scores.new_zeros((b, a), dtype=torch.bool)

        # 1. ô có tâm nằm trong gt: (B, M, A)
        in_gts = self.anchors_in_gts(anchors, gt_bboxes, mask_gt) & mask_gt.bool()

        # 2. align metric, chỉ tính cho cặp (gt, ô) hợp lệ, còn lại = 0
        cls_idx = gt_labels.long().squeeze(-1)                                    # (B, M)
        scores = pd_scores.gather(2, cls_idx[:, None, :].expand(-1, a, -1))       # (B, A, M) điểm đúng class của gt
        scores = scores.transpose(1, 2)                                           # (B, M, A)
        ious = bbox_ciou(gt_bboxes[:, :, None], pd_bboxes[:, None]).clamp(0)      # (B, M, A)
        ious = ious * in_gts
        align = scores.pow(self.alpha) * ious.pow(self.beta) * in_gts

        # 3. top-k ô theo align cho mỗi gt
        topk_idx = align.topk(self.topk, dim=-1).indices                          # (B, M, k)
        is_topk = torch.zeros_like(in_gts).scatter_(-1, topk_idx, True)
        mask_pos = (is_topk & in_gts).float()                                     # (B, M, A)

        # 4. ô bị nhiều gt chọn -> chỉ giữ gt có CIoU cao nhất
        multi = mask_pos.sum(1, keepdim=True) > 1                                 # (B, 1, A)
        best_gt = F.one_hot(ious.max(1).indices, m).transpose(1, 2).float()       # (B, M, A)
        mask_pos = torch.where(multi, best_gt, mask_pos)
        fg_mask, target_gt_idx = mask_pos.max(1)                                  # (B, A)

        # 5. gán box và class của gt cho từng ô
        target_bboxes = gt_bboxes.gather(1, target_gt_idx[..., None].expand(-1, -1, 4))  # (B, A, 4)
        target_labels = cls_idx.gather(1, target_gt_idx).clamp(0)                         # (B, A)
        target_scores = F.one_hot(target_labels, nc) * fg_mask[..., None]                 # (B, A, nc)

        # 6. nhãn mềm: align / align lớn nhất của gt * CIoU lớn nhất của gt
        align = align * mask_pos
        ious = ious * mask_pos
        norm = align * ious.amax(-1, keepdim=True) / (align.amax(-1, keepdim=True) + self.eps)  # (B, M, A)
        target_scores = target_scores * norm.amax(1)[..., None]

        return target_bboxes, target_scores, fg_mask.bool()

    def anchors_in_gts(self, anchors: torch.Tensor, gt_bboxes: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        """(B, M, A): tâm ô nằm hẳn bên trong gt. Cạnh gt < min_side được nới thành min_side quanh tâm của gt."""
        c = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2
        wh = gt_bboxes[..., 2:] - gt_bboxes[..., :2]
        wh = torch.where((wh < self.min_side) & mask_gt.bool(), torch.full_like(wh, self.min_side), wh)
        lt = (c - wh / 2)[:, :, None]                 # (B, M, 1, 2)
        rb = (c + wh / 2)[:, :, None]
        d = torch.cat([anchors - lt, rb - anchors], dim=-1)  # (B, M, A, 4)
        return d.amin(-1) > self.eps
