"""Nạp trọng số yolov8n.pt gốc vào model tự viết, so đầu ra với Ultralytics, rồi chạy trên ảnh thật.

    python -m pip install ultralytics
    python load_weights.py              # dùng ảnh bus.jpg có sẵn trong ultralytics
    python load_weights.py anh.jpg      # dùng ảnh của bạn

Kết quả vẽ box được lưu vào result.jpg.
"""
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from yolo.model import YOLOv8n
from yolo.postprocess import postprocess
from yolo.preprocess import letterbox

# Tên lớp con của mình -> chỉ số lớp trong yolov8.yaml của Ultralytics (model.0 ... model.22)
LAYERS = {
    "backbone.conv": "model.0", "backbone.conv2": "model.1", "backbone.c2f1": "model.2",
    "backbone.conv3": "model.3", "backbone.c2f2": "model.4", "backbone.conv4": "model.5",
    "backbone.c2f3": "model.6", "backbone.conv5": "model.7", "backbone.c2f4": "model.8",
    "backbone.sppf": "model.9",
    # model.10, 13 là Upsample, model.11, 14, 17, 20 là Concat: không có tham số
    "neck.c2f_t4": "model.12", "neck.c2f_n3": "model.15", "neck.down3": "model.16",
    "neck.c2f_n4": "model.18", "neck.down4": "model.19", "neck.c2f_n5": "model.21",
    "head.box_branches": "model.22.cv2", "head.cls_branches": "model.22.cv3",
}
# Tên bên trong C2f / Bottleneck / SPPF
PARTS = {"conv1": "cv1", "conv2": "cv2", "bottlenecks": "m"}


def to_ultralytics_key(key: str) -> str:
    """'backbone.c2f1.bottlenecks.0.conv1.conv.weight' -> 'model.2.m.0.cv1.conv.weight'"""
    group, name, *rest = key.split(".")
    return ".".join([LAYERS[f"{group}.{name}"]] + [PARTS.get(p, p) for p in rest])


def load_from_ultralytics(model: YOLOv8n, src: dict[str, torch.Tensor]) -> None:
    dst = {}
    for key, value in model.state_dict().items():
        if key == "head.dfl.project":  # buffer 0..15 tự tạo, bản gốc lưu dạng conv weight (model.22.dfl.conv.weight)
            continue
        src_key = to_ultralytics_key(key)
        if src_key not in src:
            raise KeyError(f"{key} -> {src_key}: không có trong yolov8n.pt")
        if src[src_key].shape != value.shape:
            raise ValueError(f"{key} -> {src_key}: shape {tuple(value.shape)}, bản gốc {tuple(src[src_key].shape)}")
        dst[key] = src[src_key]
    unused = set(src) - {to_ultralytics_key(k) for k in dst} - {"model.22.dfl.conv.weight"}
    if unused:
        raise ValueError(f"còn {len(unused)} khoá chưa dùng, ví dụ {sorted(unused)[:3]}")
    model.load_state_dict(dst, strict=False)  # strict=False chỉ vì bỏ qua head.dfl.project
    print(f"Đã nạp {len(dst)} tensor từ yolov8n.pt")


def main():
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    ref = YOLO("yolov8n.pt")  # tự tải về nếu chưa có
    names = ref.names
    ref = ref.model.float().eval()

    model = YOLOv8n().eval()
    load_from_ultralytics(model, ref.state_dict())

    # 1. So đầu ra thô trên cùng một tensor đầu vào
    path = sys.argv[1] if len(sys.argv) > 1 else str(ASSETS / "bus.jpg")
    image = np.array(Image.open(path).convert("RGB"))
    x, scale, (pad_x, pad_y) = letterbox(image)
    with torch.no_grad():
        mine = model(x)
        theirs = ref(x)[0]  # Detect ở chế độ eval trả về (y, x); y có shape (1, 84, 8400)
    diff = (mine - theirs).abs().max().item()
    print(f"Sai khác lớn nhất so với Ultralytics: {diff:.2e}  ({'KHỚP' if diff < 1e-3 else 'LỆCH'})")

    # 2. Lọc box, đổi về toạ độ ảnh gốc, vẽ ra ảnh
    dets = postprocess(mine)
    dets[:, [0, 2]] = ((dets[:, [0, 2]] - pad_x) / scale).clamp(0, image.shape[1])
    dets[:, [1, 3]] = ((dets[:, [1, 3]] - pad_y) / scale).clamp(0, image.shape[0])

    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    for x1, y1, x2, y2, score, cls in dets.tolist():
        label = f"{names[int(cls)]} {score:.2f}"
        print(f"  {label:20s} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1 + 3, y1 + 2), label, fill=(255, 0, 0))
    canvas.save("result.jpg")
    print(f"{len(dets)} box, đã lưu result.jpg")


if __name__ == "__main__":
    main()
