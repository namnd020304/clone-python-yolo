"""Train YOLOv8n gốc của Ultralytics từ đầu, cùng thông số với train.py, để so với model tự viết.

    python train_ultralytics.py --data ../data/coco --train-split train2017 --val-split val2017 --epochs 30 --batch 16 --workers 8 --lr 0.001 --out runs/ultra-coco

Kết quả: <out>/weights/best.pt, last.pt; mỗi epoch ghi P, R, mAP50, mAP50-95, loss vào <out>/results.csv (kèm biểu đồ).
Chạy tiếp khi bị ngắt: python train_ultralytics.py --resume <out>/weights/last.pt
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from yolo.dataset import COCO_NAMES, download_coco8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="coco8")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--lr", type=float, default=0.0, help="0 = 0.002 * 5 / (4 + nc) như train.py")
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="bật mixed precision (train.py chạy fp32 nên mặc định tắt)")
    ap.add_argument("--out", default="runs/ultra")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    if args.resume:
        YOLO(args.resume).train(resume=True)
        return

    # Ultralytics cần file data.yaml; nhãn tự tìm bằng cách đổi /images/ thành /labels/, khớp cấu trúc của train.py
    data = (download_coco8() if args.data == "coco8" else Path(args.data)).resolve()
    names_file = data / "classes.txt"
    names = ([n.strip() for n in names_file.read_text(encoding="utf-8").splitlines() if n.strip()]
             if names_file.exists() else COCO_NAMES)
    lr = args.lr or round(0.002 * 5 / (4 + len(names)), 6)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(data), "train": f"images/{args.train_split}",
                                         "val": f"images/{args.val_split}", "names": dict(enumerate(names))},
                                        allow_unicode=True), encoding="utf-8")

    model = YOLO("yolov8n.yaml")  # .yaml = kiến trúc, trọng số ngẫu nhiên; .pt mới là trọng số đã train
    model.train(
        data=str(data_yaml), epochs=args.epochs, batch=args.batch, imgsz=args.size, workers=args.workers,
        project=str(out.parent), name=out.name, exist_ok=True, pretrained=False, seed=0, amp=args.amp,
        # optimizer + lịch lr giống train.py: AdamW, warmup 3 epoch từ 0, cosine về lr * 0.01
        optimizer="AdamW", lr0=lr, lrf=0.01, cos_lr=True, momentum=0.9, weight_decay=args.weight_decay,
        warmup_epochs=3, warmup_bias_lr=0.0, warmup_momentum=0.9,
        nbs=args.batch,  # mặc định nbs=64 sẽ cộng dồn gradient và scale weight decay; đặt = batch để tắt
        # augmentation giống train.py: mosaic (zoom 0.5..1.5, lệch ±10%), lật ngang, sáng/tối ±25%
        mosaic=1.0, close_mosaic=args.close_mosaic, scale=0.5, translate=0.1, fliplr=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.25, degrees=0.0, shear=0.0, perspective=0.0, mixup=0.0,
        # loss gain giống yolo/loss.py
        box=7.5, cls=0.5, dfl=1.5,
    )
    model.val(data=str(data_yaml), imgsz=args.size, batch=args.batch, conf=0.001, iou=0.7)


if __name__ == "__main__":
    main()
