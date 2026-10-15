"""Bước 14: train YOLOv8n tự viết.

    python train.py --epochs 150 --val-split train            # train từ đầu, overfit 4 ảnh train của coco8
    python train.py --weights yolov8n.pt --epochs 30           # fine-tune từ trọng số gốc (cần ultralytics)
    python train.py --data datasets/bien-bao --weights yolov8n.pt   # dataset riêng (tạo bằng kaggle_data.py)

--data là thư mục có images/<split> và labels/<split>; để "coco8" thì tự tải về datasets/coco8.
Nếu thư mục có classes.txt (mỗi dòng một tên class) thì số class và tên class lấy từ đó, không cần --nc.
Checkpoint lưu vào --out (last.pt, best.pt) là trọng số EMA; xem kết quả bằng predict.py.
Augmentation khi train: mosaic (tắt ở --close-mosaic epoch cuối), lật ngang, đổi độ sáng; --no-augment tắt hết.
"""
import argparse
import sys
import math
import time
from pathlib import Path

import torch
from torch import nn

from yolo.dataset import COCO_NAMES, YoloDataset, collate, download_coco8
from yolo.ema import ModelEMA
from yolo.loss import DetectionLoss
from yolo.metrics import evaluate
from yolo.model import YOLOv8n


def load_pretrained(model: YOLOv8n, path: str) -> None:
    """Nạp checkpoint của train.py, hoặc yolov8n.pt của Ultralytics. Tensor khác shape (ví dụ lớp class cuối khi số
    class khác 80) được bỏ qua và giữ giá trị khởi tạo."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    src = ckpt["model"]
    if isinstance(src, nn.Module):  # file của Ultralytics: đổi tên khoá như load_weights.py
        from load_weights import to_ultralytics_key
        u = src.float().state_dict()
        src = {k: u[to_ultralytics_key(k)] for k in model.state_dict() if k != "head.dfl.project"}
    own = model.state_dict()
    ok = {k: v for k, v in src.items() if k in own and v.shape == own[k].shape}
    model.load_state_dict(ok, strict=False)
    print(f"Nạp {len(ok)}/{len(own)} tensor từ {path}")


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW; chỉ trọng số conv có weight decay, BatchNorm và bias thì không (giống Ultralytics)."""
    decay, no_decay = [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            (decay if name == "weight" and not isinstance(module, nn.BatchNorm2d) else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.999))


def lr_at(it: int, total: int, warmup: int, lr: float, final_ratio: float = 0.01) -> float:
    """Tăng tuyến tính trong warmup, sau đó giảm theo cosine về lr * final_ratio."""
    if it < warmup:
        return lr * (it + 1) / warmup
    t = (it - warmup) / max(1, total - warmup)
    return lr * (final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="coco8")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--nc", type=int, default=0, help="số class; 0 = đọc từ classes.txt, không có thì 80")
    ap.add_argument("--weights", default="", help="yolov8n.pt hoặc checkpoint của train.py; bỏ trống = train từ đầu")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--lr", type=float, default=0.0, help="0 = tự chọn 0.002 * 5 / (4 + nc) như Ultralytics")
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--mosaic", type=float, default=1.0, help="xác suất dùng mosaic cho mỗi ảnh train")
    ap.add_argument("--close-mosaic", type=int, default=10, help="tắt mosaic ở N epoch cuối")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="runs/train")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = download_coco8() if args.data == "coco8" else Path(args.data)
    names_file = data / "classes.txt"
    if names_file.exists():
        names = [n.strip() for n in names_file.read_text(encoding="utf-8").splitlines() if n.strip()]
        args.nc = args.nc or len(names)
    else:
        args.nc = args.nc or 80
        names = COCO_NAMES if args.nc == 80 else []
    names += [str(i) for i in range(len(names), args.nc)]
    if args.lr <= 0:
        args.lr = round(0.002 * 5 / (4 + args.nc), 6)  # 80 class -> 0.000119; ít class -> lr lớn hơn
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_set = YoloDataset(data / "images" / args.train_split, args.size, augment=not args.no_augment,
                            mosaic=args.mosaic)
    val_set = YoloDataset(data / "images" / args.val_split, args.size)
    train_loader = torch.utils.data.DataLoader(train_set, args.batch, shuffle=True, collate_fn=collate,
                                               num_workers=args.workers, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_set, args.batch, collate_fn=collate, num_workers=args.workers)

    model = YOLOv8n(args.nc).to(device)
    if args.weights:
        load_pretrained(model, args.weights)
    ema = None if args.no_ema else ModelEMA(model)
    criterion = DetectionLoss(num_classes=args.nc)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    total = args.epochs * len(train_loader)
    warmup = min(max(3 * len(train_loader), 10), total // 5)
    print(f"{len(train_set)} ảnh train, {len(val_set)} ảnh val, {args.nc} class, {total} vòng lặp, lr {args.lr}, "
          f"thiết bị {device}")
    if args.weights:
        m50, m = evaluate(model, val_loader)
        print(f"Trước khi train: mAP50 {m50:.3f}  mAP50-95 {m:.3f}")

    best, it = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if train_set.mosaic and epoch > args.epochs - args.close_mosaic:
            train_set.mosaic = 0.0
            print(f"Tắt mosaic từ epoch {epoch}")
        t0, sums = time.time(), torch.zeros(3)
        for imgs, gt_labels, gt_bboxes, mask in train_loader:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(it, total, warmup, args.lr)
            imgs, gt_labels, gt_bboxes, mask = (t.to(device) for t in (imgs, gt_labels, gt_bboxes, mask))
            loss = criterion(model(imgs), gt_labels, gt_bboxes, mask)       # [box, cls, dfl]
            optimizer.zero_grad()
            (loss.sum() * len(imgs)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            if ema:
                ema.update(model)
            sums += loss.detach().cpu()
            it += 1
        box, cls, dfl = (sums / len(train_loader)).tolist()
        line = (f"epoch {epoch:4d}/{args.epochs}  box {box:.3f}  cls {cls:.3f}  dfl {dfl:.3f}  "
                f"lr {optimizer.param_groups[0]['lr']:.5f}  {time.time() - t0:.1f}s")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            final = ema.ema if ema else model
            m50, m = evaluate(final, val_loader)
            line += f"  |  mAP50 {m50:.3f}  mAP50-95 {m:.3f}"
            ckpt = {"model": final.state_dict(), "epoch": epoch, "nc": args.nc, "names": names, "map": m}
            torch.save(ckpt, out / "last.pt")
            if m > best:
                best = m
                torch.save(ckpt, out / "best.pt")
        print(line)
    print(f"Xong. mAP50-95 tốt nhất {best:.3f}, checkpoint ở {out}")


if __name__ == "__main__":
    main()
