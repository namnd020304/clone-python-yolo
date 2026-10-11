"""Chạy checkpoint của train.py trên một ảnh, vẽ box ra file. Không cần ultralytics.

    python predict.py runs/train/best.pt datasets/coco8/images/val/000000000036.jpg
"""
import argparse
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from yolo.model import YOLOv8n
from yolo.postprocess import postprocess
from yolo.preprocess import letterbox


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("image")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="predict.jpg")
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = YOLOv8n(ckpt["nc"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    names = ckpt["names"]

    image = np.array(Image.open(args.image).convert("RGB"))
    x, scale, (pad_x, pad_y) = letterbox(image)
    with torch.no_grad():
        dets = postprocess(model(x), args.conf)
    dets[:, [0, 2]] = ((dets[:, [0, 2]] - pad_x) / scale).clamp(0, image.shape[1])
    dets[:, [1, 3]] = ((dets[:, [1, 3]] - pad_y) / scale).clamp(0, image.shape[0])

    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    for x1, y1, x2, y2, score, cls in dets.tolist():
        label = f"{names[int(cls)]} {score:.2f}"
        print(f"  {label:20s} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1 + 3, y1 + 2), label, fill=(255, 0, 0))
    canvas.save(args.out)
    print(f"{len(dets)} box, đã lưu {args.out}")


if __name__ == "__main__":
    main()
