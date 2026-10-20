"""So best.pt (tự train) với yolov8n.pt gốc trên cùng ảnh, cùng code repo, cùng ngưỡng."""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from load_weights import load_from_ultralytics
from yolo.model import YOLOv8n
from yolo.postprocess import postprocess
from yolo.preprocess import letterbox

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "images")
torch.set_num_threads(os.cpu_count())


def load_mine(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = YOLOv8n(ckpt["nc"])
    m.load_state_dict(ckpt["model"])
    return m.eval(), ckpt["names"]


def load_ref(path):
    from ultralytics import YOLO
    ref = YOLO(path)
    m = YOLOv8n().eval()
    load_from_ultralytics(m, ref.model.float().state_dict())
    return m, ref.names


def run(model, image, conf):
    x, scale, (px, py) = letterbox(image)
    with torch.no_grad():
        dets = postprocess(model(x), conf)
    dets[:, [0, 2]] = ((dets[:, [0, 2]] - px) / scale).clamp(0, image.shape[1])
    dets[:, [1, 3]] = ((dets[:, [1, 3]] - py) / scale).clamp(0, image.shape[0])
    return dets


def speed(model, image, n=30):
    x, _, _ = letterbox(image)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t = time.perf_counter()
        for _ in range(n):
            model(x)
    return (time.perf_counter() - t) / n * 1000


def draw(image, dets, names, title):
    c = Image.fromarray(image)
    d = ImageDraw.Draw(c)
    for x1, y1, x2, y2, s, k in dets.tolist():
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        d.text((x1 + 3, y1 + 2), f"{names[int(k)]} {s:.2f}", fill=(255, 0, 0))
    d.rectangle([0, 0, 260, 22], fill=(0, 0, 0))
    d.text((6, 5), title, fill=(255, 255, 255))
    return c


def main():
    best, ref_path, *images = sys.argv[1:]
    models = {"best.pt": load_mine(best), "yolov8n.pt": load_ref(ref_path)}
    for img_path in images:
        image = np.array(Image.open(img_path).convert("RGB"))
        name = os.path.splitext(os.path.basename(img_path))[0]
        panels = []
        print(f"\n### {name}")
        for tag, (m, names) in models.items():
            dets = run(m, image, 0.25)
            print(f"{tag}: {len(dets)} box")
            for x1, y1, x2, y2, s, k in dets.tolist():
                print(f"  {names[int(k)]:14s} {s:.2f}  [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            panels.append(draw(image, dets, names, tag))
        w, h = panels[0].size
        out = Image.new("RGB", (w * 2 + 10, h), (255, 255, 255))
        out.paste(panels[0], (0, 0))
        out.paste(panels[1], (w + 10, 0))
        out.save(os.path.join(OUT, f"so-sanh-{name}.jpg"))
    image = np.array(Image.open(images[0]).convert("RGB"))
    for tag, (m, _) in models.items():
        print(f"tốc độ CPU {tag}: {speed(m, image):.1f} ms/ảnh 640")
        print(f"  tham số: {sum(p.numel() for p in m.parameters()):,}")


if __name__ == "__main__":
    main()
