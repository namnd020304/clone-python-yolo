"""Chương trình chấm clone-yolo.

    python check.py        chấm tất cả bước
    python check.py 4      chỉ chấm bước 4
"""
import sys

import numpy as np
import torch

torch.manual_seed(0)


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def expect(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def expect_shape(t: torch.Tensor, shape: tuple, name: str):
    expect(tuple(t.shape) == shape, f"{name}: shape {tuple(t.shape)}, cần {shape}")


def step1():
    from yolo.blocks import Conv
    m = Conv(3, 16, k=3, s=2)
    expect_shape(m(torch.randn(1, 3, 640, 640)), (1, 16, 320, 320), "Conv(3,16,3,2)")
    expect_shape(Conv(16, 32, k=1)(torch.randn(1, 16, 40, 40)), (1, 32, 40, 40), "Conv(16,32,1)")
    expect(any(isinstance(x, torch.nn.BatchNorm2d) for x in m.modules()), "Conv thiếu BatchNorm2d")
    expect(any(isinstance(x, torch.nn.SiLU) for x in m.modules()), "Conv thiếu SiLU")
    expect(n_params(m) == 464, f"Conv(3,16,3,2) có {n_params(m)} tham số, cần 464 (Conv2d không bias + BN)")


def step2():
    from yolo.blocks import C2f, Bottleneck
    expect_shape(Bottleneck(16)(torch.randn(1, 16, 40, 40)), (1, 16, 40, 40), "Bottleneck(16)")
    expect_shape(C2f(32, 64, n=2)(torch.randn(1, 32, 80, 80)), (1, 64, 80, 80), "C2f(32,64,n=2)")
    m = C2f(32, 32, n=1, shortcut=True)
    expect(n_params(m) == 7360, f"C2f(32,32,n=1) có {n_params(m)} tham số, cần 7360")


def step3():
    from yolo.blocks import SPPF
    m = SPPF(256, 256)
    expect_shape(m(torch.randn(1, 256, 20, 20)), (1, 256, 20, 20), "SPPF(256,256)")
    expect(n_params(m) == 164608, f"SPPF(256,256) có {n_params(m)} tham số, cần 164608")


def step4():
    from yolo.backbone import Backbone
    p3, p4, p5 = Backbone()(torch.randn(1, 3, 640, 640))
    expect_shape(p3, (1, 64, 80, 80), "P3")
    expect_shape(p4, (1, 128, 40, 40), "P4")
    expect_shape(p5, (1, 256, 20, 20), "P5")


def step5():
    from yolo.neck import Neck
    n3, n4, n5 = Neck()(torch.randn(1, 64, 80, 80), torch.randn(1, 128, 40, 40), torch.randn(1, 256, 20, 20))
    expect_shape(n3, (1, 64, 80, 80), "N3")
    expect_shape(n4, (1, 128, 40, 40), "N4")
    expect_shape(n5, (1, 256, 20, 20), "N5")


def step6():
    from yolo.head import DFL, Head
    d = DFL()(torch.zeros(1, 64, 10))
    expect_shape(d, (1, 4, 10), "DFL")
    expect(torch.allclose(d, torch.full_like(d, 7.5)), "DFL: logit toàn 0 thì khoảng cách phải là 7.5")
    head = Head().eval()
    with torch.no_grad():
        out = head([torch.randn(1, 64, 80, 80), torch.randn(1, 128, 40, 40), torch.randn(1, 256, 20, 20)])
    expect_shape(out, (1, 84, 8400), "Head")
    cls = out[:, 4:]
    expect(bool(((cls >= 0) & (cls <= 1)).all()), "Head: điểm class phải qua sigmoid (nằm trong 0..1)")
    expect(bool((out[:, 2:4] >= 0).all()), "Head: w, h không được âm")


def step7():
    from yolo.model import YOLOv8n
    m = YOLOv8n().eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 640, 640))
    expect_shape(out, (1, 84, 8400), "YOLOv8n")
    n = n_params(m)
    expect(3_150_000 <= n <= 3_160_000, f"YOLOv8n có {n:,} tham số, bản gốc khoảng 3,157,200")


def step8():
    from yolo.preprocess import letterbox
    img = np.full((480, 640, 3), 255, dtype=np.uint8)
    t, scale, (px, py) = letterbox(img)
    expect_shape(t, (1, 3, 640, 640), "letterbox(480x640)")
    expect(t.dtype == torch.float32, f"letterbox: dtype {t.dtype}, cần float32")
    expect(abs(scale - 1.0) < 1e-6 and px == 0 and py == 80, f"letterbox(480x640): scale={scale}, pad=({px},{py}), cần 1.0, (0, 80)")
    expect(abs(t[0, 0, 0, 0].item() - 114 / 255) < 1e-3, "letterbox: viền phải là 114/255")
    expect(abs(t[0, 0, 320, 320].item() - 1.0) < 1e-3, "letterbox: giữa ảnh trắng phải là 1.0 (chia 255)")
    t, scale, (px, py) = letterbox(np.zeros((720, 1280, 3), dtype=np.uint8))
    expect(abs(scale - 0.5) < 1e-6 and px == 0 and py == 140, f"letterbox(720x1280): scale={scale}, pad=({px},{py}), cần 0.5, (0, 140)")


def step9():
    import torchvision
    from yolo.postprocess import box_iou, nms, postprocess
    a = torch.tensor([[0., 0., 10., 10.]])
    b = torch.tensor([[0., 0., 10., 10.], [5., 0., 15., 10.], [20., 20., 30., 30.]])
    iou = box_iou(a, b)
    expect_shape(iou, (1, 3), "box_iou")
    expect(torch.allclose(iou, torch.tensor([[1.0, 1 / 3, 0.0]]), atol=1e-4), f"box_iou sai: {iou.tolist()}")

    xy = torch.rand(200, 2) * 500
    boxes = torch.cat([xy, xy + torch.rand(200, 2) * 100 + 1], dim=1)
    scores = torch.rand(200)
    mine = set(nms(boxes, scores, 0.5).tolist())
    ref = set(torchvision.ops.nms(boxes, scores, 0.5).tolist())
    expect(mine == ref, f"nms: giữ {len(mine)} box, đáp án giữ {len(ref)} box")

    pred = torch.zeros(1, 84, 8400)
    pred[0, :4, 0] = torch.tensor([100., 100., 50., 50.])   # class 0, điểm 0.9
    pred[0, 4, 0] = 0.9
    pred[0, :4, 1] = torch.tensor([102., 100., 50., 50.])   # trùng box trên, điểm thấp hơn -> bị NMS bỏ
    pred[0, 4, 1] = 0.8
    pred[0, :4, 2] = torch.tensor([102., 100., 50., 50.])   # cùng vị trí nhưng class 2 -> được giữ
    pred[0, 6, 2] = 0.7
    pred[0, 4, 3] = 0.1                                      # dưới ngưỡng 0.25 -> bị bỏ
    out = postprocess(pred)
    expect_shape(out, (2, 6), "postprocess")
    expect(torch.allclose(out[0], torch.tensor([75., 75., 125., 125., 0.9, 0.])), f"postprocess: box đầu sai {out[0].tolist()}")


def step10():
    import tempfile
    from pathlib import Path
    from PIL import Image
    from yolo.dataset import YoloDataset, collate
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "images").mkdir()
        (Path(d) / "labels").mkdir()
        Image.new("RGB", (640, 480)).save(Path(d) / "images" / "a.jpg")
        (Path(d) / "labels" / "a.txt").write_text("3 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.1 0.1\n")
        Image.new("RGB", (640, 480)).save(Path(d) / "images" / "b.jpg")  # ảnh không có nhãn
        ds = YoloDataset(Path(d) / "images")
        img, lab = ds[0]
        expect_shape(img, (3, 640, 640), "ảnh")
        expect(torch.allclose(lab[0], torch.tensor([3., 160., 200., 480., 440.])),
               f"nhãn sau letterbox sai: {lab[0].tolist()}, cần [3, 160, 200, 480, 440] (ảnh 480x640 có viền trên 80)")
        imgs, cls, boxes, mask = collate([ds[0], ds[1]])
        expect_shape(imgs, (2, 3, 640, 640), "collate: ảnh")
        expect_shape(boxes, (2, 2, 4), "collate: box")
        expect(mask[:, :, 0].tolist() == [[1, 1], [0, 0]], f"collate: mask sai {mask[:, :, 0].tolist()}")


def step11():
    from yolo.model import YOLOv8n
    box, cls, anchors, strides = YOLOv8n().train()(torch.randn(2, 3, 640, 640))
    expect_shape(box, (2, 64, 8400), "train: box")
    expect_shape(cls, (2, 80, 8400), "train: cls")
    expect_shape(anchors, (8400, 2), "anchors")
    expect(strides[:6400].eq(8).all() and strides[-400:].eq(32).all(), "strides: 6400 ô đầu stride 8, 400 ô cuối stride 32")


def random_targets():
    g = torch.Generator().manual_seed(1)
    xy = torch.rand(2, 5, 2, generator=g) * 500
    gt_bboxes = torch.cat([xy, xy + torch.rand(2, 5, 2, generator=g) * 200 + 4], dim=-1)
    gt_labels = torch.randint(0, 80, (2, 5, 1), generator=g).float()
    mask_gt = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]).float()[..., None]
    return gt_labels, gt_bboxes * mask_gt, mask_gt


def step12():
    from ultralytics.utils.tal import TaskAlignedAssigner as Ref
    from yolo.assigner import TaskAlignedAssigner
    from yolo.head import make_anchors
    anchors, strides = make_anchors([torch.zeros(1, 1, 80, 80), torch.zeros(1, 1, 40, 40), torch.zeros(1, 1, 20, 20)],
                                    (8, 16, 32))
    anchors = anchors * strides
    g = torch.Generator().manual_seed(2)
    scores = torch.rand(2, 8400, 80, generator=g)
    c = anchors.expand(2, -1, -1)
    pd = torch.cat([c - torch.rand(2, 8400, 2, generator=g) * 60, c + torch.rand(2, 8400, 2, generator=g) * 60], -1)
    gt_labels, gt_bboxes, mask_gt = random_targets()
    tb, ts, fg = TaskAlignedAssigner()(scores, pd, anchors, gt_labels, gt_bboxes, mask_gt)
    _, rtb, rts, rfg, _ = Ref(topk=10, num_classes=80, alpha=0.5, beta=6.0, stride=[8, 16, 32])(
        scores, pd, anchors, gt_labels, gt_bboxes, mask_gt)
    expect(bool((fg == rfg).all()), f"fg_mask: {int(fg.sum())} ô dương, Ultralytics {int(rfg.sum())}")
    expect(torch.allclose(tb[fg], rtb[fg]), "target_bboxes khác Ultralytics")
    expect(torch.allclose(ts, rts.float(), atol=1e-6), "target_scores khác Ultralytics")


def step13():
    from types import SimpleNamespace
    from ultralytics.utils.loss import v8DetectionLoss
    from yolo.loss import DetectionLoss, dfl_loss
    from yolo.model import YOLOv8n
    # khoảng cách thật 3.3 -> loss nhỏ nhất khi bin 3 có xác suất 0.7, bin 4 có 0.3: loss = -(0.7 ln 0.7 + 0.3 ln 0.3)
    best = torch.log(torch.tensor([0.0] * 3 + [0.7, 0.3] + [0.0] * 11) + 1e-30).expand(1, 4, 16)
    got = dfl_loss(best, torch.tensor([[3.3] * 4])).item()
    expect(abs(got - 0.6109) < 1e-3, f"dfl_loss(phân bố 0.7/0.3, đích 3.3) = {got:.4f}, cần 0.6109")
    model = YOLOv8n().train()
    preds = model(torch.randn(2, 3, 640, 640))
    gt_labels, gt_bboxes, mask_gt = random_targets()
    mine = DetectionLoss()(preds, gt_labels, gt_bboxes, mask_gt)

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
            self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
            self.model = [SimpleNamespace(stride=torch.tensor([8., 16., 32.]), nc=80, reg_max=16)]

    keep = mask_gt[..., 0].bool()
    xyxy = gt_bboxes[keep] / 640
    batch = {"batch_idx": keep.nonzero()[:, 0].float(), "cls": gt_labels[keep][:, 0],
             "bboxes": torch.cat([(xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]], 1)}
    feats = [torch.zeros(2, 1, 80, 80), torch.zeros(2, 1, 40, 40), torch.zeros(2, 1, 20, 20)]
    ref = v8DetectionLoss(Fake()).get_assigned_targets_and_loss({"boxes": preds[0], "scores": preds[1], "feats": feats},
                                                                batch)[1]
    expect(torch.allclose(mine, ref, rtol=1e-4), f"loss [box, cls, dfl] = {mine.tolist()}, Ultralytics {ref.tolist()}")
    mine.sum().backward()
    expect(model.head.cls_branches[0][-1].weight.grad is not None, "loss không lan ngược (backward) về model")


def step14():
    from train import build_optimizer
    from yolo.loss import DetectionLoss
    from yolo.model import YOLOv8n
    torch.manual_seed(0)
    model = YOLOv8n().train()
    img = torch.rand(1, 3, 320, 320)
    gt_labels, gt_bboxes, mask = torch.tensor([[[5.]]]), torch.tensor([[[60., 80., 220., 260.]]]), torch.ones(1, 1, 1)
    opt = build_optimizer(model, 0.002, 5e-4)
    crit = DetectionLoss()
    losses = []
    for _ in range(15):
        loss = crit(model(img), gt_labels, gt_bboxes, mask).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    expect(losses[-1] < 0.5 * losses[0], f"15 bước train trên 1 ảnh: loss {losses[0]:.2f} -> {losses[-1]:.2f}, cần giảm hơn nửa")


def step15():
    from yolo.metrics import IOU_THRESHOLDS, match_predictions, mean_ap
    gt = torch.tensor([[0., 0., 0., 100., 100.], [1., 200., 200., 300., 300.]])
    perfect = torch.tensor([[0., 0., 100., 100., 0.9, 0.], [200., 200., 300., 300., 0.8, 1.]])
    tp = match_predictions(perfect, gt)
    expect_shape(tp, (2, len(IOU_THRESHOLDS)), "match_predictions")
    m50, m = mean_ap(tp, perfect[:, 4], perfect[:, 5], gt[:, 0])
    # nội suy 101 điểm + hình thang nên tối đa là 0.995 chứ không phải 1 (Ultralytics cũng vậy)
    expect(abs(m50 - 0.995) < 1e-6 and abs(m - 0.995) < 1e-6, f"dự đoán hoàn hảo phải cho mAP = 0.995, được {m50:.3f}, {m:.3f}")
    wrong = perfect.clone()
    wrong[1, 5] = 0.  # đúng chỗ nhưng sai class
    tp = match_predictions(wrong, gt)
    expect(tp[1].sum() == 0, "box sai class không được tính là đúng")
    m50, _ = mean_ap(tp, wrong[:, 4], wrong[:, 5], gt[:, 0])
    expect(abs(m50 - 0.995 / 2) < 1e-6, f"1/2 class đúng hoàn toàn phải cho mAP50 = 0.4975, được {m50:.4f}")
    shifted = torch.tensor([[0., 30., 100., 130., 0.9, 0.]])  # IoU = 70 / 130 ~ 0.54
    tp = match_predictions(shifted, gt[:1])
    expect(tp[0, 0] and not tp[0, 1], "IoU 0.54: đúng ở ngưỡng 0.5, sai ở ngưỡng 0.55")


STEPS = [step1, step2, step3, step4, step5, step6, step7, step8, step9, step10, step11, step12, step13, step14, step15]
NAMES = ["Conv", "Bottleneck + C2f", "SPPF", "Backbone", "Neck", "Head + DFL", "YOLOv8n", "letterbox",
         "IoU + NMS + postprocess", "dataset + collate", "Head trả đầu ra thô khi train", "TAL assigner (so Ultralytics)",
         "loss (so Ultralytics)", "train vài bước, loss giảm", "mAP"]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    chosen = [int(a) for a in sys.argv[1:]] or range(1, len(STEPS) + 1)
    passed = 0
    for i in chosen:
        try:
            STEPS[i - 1]()
            print(f"✓ {i}. {NAMES[i - 1]}")
            passed += 1
        except NotImplementedError:
            print(f"· {i}. {NAMES[i - 1]}: chưa làm")
        except Exception as e:  # lỗi trong code của bạn: in ra để sửa
            print(f"✗ {i}. {NAMES[i - 1]}: {type(e).__name__}: {e}")
    print(f"\nKết quả: {passed}/{len(chosen)}")


if __name__ == "__main__":
    main()
