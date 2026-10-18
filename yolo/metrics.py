"""Bước 15: đo chất lượng bằng mAP (mean Average Precision), cách tính giống COCO / Ultralytics."""
import numpy as np
import torch

from .postprocess import box_iou

IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)  # 0.50, 0.55, ..., 0.95


def match_predictions(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Đánh dấu dự đoán nào đúng (true positive) ở từng ngưỡng IoU.

    pred (K, 6) = [x1, y1, x2, y2, score, class], gt (G, 5) = [class, x1, y1, x2, y2].
    Một dự đoán đúng khi cùng class với một gt và IoU >= ngưỡng; mỗi gt chỉ được ghép với một dự đoán, ưu tiên cặp
    có IoU cao hơn. Trả về (K, 10) bool.
    """
    tp = torch.zeros(len(pred), len(IOU_THRESHOLDS), dtype=torch.bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp
    iou = box_iou(gt[:, 1:], pred[:, :4]) * (gt[:, :1] == pred[None, :, 5])  # (G, K), khác class -> 0
    for j, t in enumerate(IOU_THRESHOLDS):
        gi, pi = torch.nonzero(iou >= t, as_tuple=True)
        if len(gi) == 0:
            continue
        order = iou[gi, pi].argsort(descending=True)                          # cặp IoU cao ghép trước
        gi, pi = gi[order].numpy(), pi[order].numpy()
        _, first = np.unique(pi, return_index=True)                          # mỗi dự đoán một gt
        gi, pi = gi[first], pi[first]
        order = iou[gi, pi].argsort(descending=True).numpy()
        gi, pi = gi[order], pi[order]
        _, first = np.unique(gi, return_index=True)                          # mỗi gt một dự đoán
        tp[pi[first], j] = True
    return tp


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Diện tích dưới đường precision-recall, nội suy 101 điểm như COCO."""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))  # precision không tăng khi recall tăng
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def mean_ap(tp: torch.Tensor, conf: torch.Tensor, pred_cls: torch.Tensor, gt_cls: torch.Tensor) -> tuple[float, float]:
    """Gộp kết quả của mọi ảnh rồi tính mAP. tp (N, 10), conf (N,), pred_cls (N,), gt_cls (G,).

    Với mỗi class có trong gt: xếp dự đoán theo điểm giảm dần, cộng dồn TP / FP ra đường precision-recall, lấy AP.
    mAP = trung bình AP trên các class. Trả về (mAP@0.5, mAP@0.5:0.95).
    """
    tp, conf, pred_cls, gt_cls = tp.numpy(), conf.numpy(), pred_cls.numpy(), gt_cls.numpy()
    order = np.argsort(-conf, kind="stable")
    tp, pred_cls = tp[order], pred_cls[order]
    classes = np.unique(gt_cls)
    if len(classes) == 0:
        return 0.0, 0.0
    ap = np.zeros((len(classes), tp.shape[1]))
    for ci, c in enumerate(classes):
        sel = pred_cls == c
        n_gt = (gt_cls == c).sum()
        if sel.sum() == 0:
            continue
        tpc = tp[sel].cumsum(0)
        fpc = (1 - tp[sel]).cumsum(0)
        recall = tpc / (n_gt + 1e-16)
        precision = tpc / (tpc + fpc)
        for j in range(tp.shape[1]):
            ap[ci, j] = average_precision(recall[:, j], precision[:, j])
    return float(ap[:, 0].mean()), float(ap.mean())


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, conf_thres: float = 0.001, iou_thres: float = 0.7) -> tuple[float, float]:
    """Chạy model trên cả loader (ngưỡng điểm rất thấp để có đủ đường precision-recall), trả về mAP50, mAP50-95."""
    from tqdm import tqdm

    from .postprocess import postprocess

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    tps, confs, pcls, gcls = [], [], [], []
    for imgs, gt_labels, gt_bboxes, mask in tqdm(loader, desc="val", leave=False, dynamic_ncols=True):
        out = model(imgs.to(device)).cpu()
        for i in range(len(imgs)):
            pred = postprocess(out[i:i + 1], conf_thres, iou_thres)[:300]
            keep = mask[i, :, 0].bool()
            gt = torch.cat([gt_labels[i][keep], gt_bboxes[i][keep]], dim=1)
            tps.append(match_predictions(pred, gt))
            confs.append(pred[:, 4])
            pcls.append(pred[:, 5])
            gcls.append(gt[:, 0])
    model.train(was_training)
    return mean_ap(torch.cat(tps), torch.cat(confs), torch.cat(pcls), torch.cat(gcls))
