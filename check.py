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


STEPS = [step1, step2, step3, step4, step5, step6, step7, step8, step9]
NAMES = ["Conv", "Bottleneck + C2f", "SPPF", "Backbone", "Neck", "Head + DFL", "YOLOv8n", "letterbox",
         "IoU + NMS + postprocess"]


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
