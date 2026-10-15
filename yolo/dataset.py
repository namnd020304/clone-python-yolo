"""Bước 10: đọc dataset định dạng YOLO.

Cấu trúc thư mục (giống coco8 của Ultralytics):

    coco8/images/train/000000000009.jpg
    coco8/labels/train/000000000009.txt     mỗi dòng: class cx cy w h  (chia cho rộng / cao ảnh, nằm trong 0..1)
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .preprocess import letterbox

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
COCO8_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def download_coco8(root: str | Path = "datasets") -> Path:
    """Tải coco8 (4 ảnh train + 4 ảnh val, 80 class COCO) nếu chưa có. Trả về thư mục coco8."""
    root = Path(root)
    if not (root / "coco8").exists():
        print(f"Đang tải {COCO8_URL} ...")
        with urllib.request.urlopen(COCO8_URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(root)
    return root / "coco8"


class YoloDataset(torch.utils.data.Dataset):
    """Trả về (ảnh (3, size, size) float 0..1, nhãn (n, 5) = [class, x1, y1, x2, y2] pixel trong ảnh size x size).

    augment=True:
      - mosaic với xác suất self.mosaic (mặc định 1.0): ghép 4 ảnh quanh một điểm chung, zoom ngẫu nhiên x0.5..1.5
        (xem load_mosaic). train.py tắt mosaic (đặt self.mosaic = 0) ở vài epoch cuối.
      - lật ngang ngẫu nhiên và đổi độ sáng nhẹ.
    augment=False: chỉ letterbox (dùng cho tập val).
    """

    def __init__(self, img_dir: str | Path, size: int = 640, augment: bool = False, mosaic: float = 1.0):
        self.files = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
        if not self.files:
            raise FileNotFoundError(f"không có ảnh trong {img_dir}")
        self.size = size
        self.augment = augment
        self.mosaic = mosaic if augment else 0.0

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def label_path(img_path: Path) -> Path:
        parts = list(img_path.parts)
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"  # thư mục "images" cuối cùng -> "labels"
        return Path(*parts).with_suffix(".txt")

    def load_labels(self, img_path: Path) -> np.ndarray:
        p = self.label_path(img_path)
        if not p.exists() or p.stat().st_size == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.loadtxt(p, dtype=np.float32, ndmin=2)[:, :5]

    def load_scaled(self, i: int, long_side: int) -> tuple[np.ndarray, torch.Tensor]:
        """Ảnh thứ i co về cạnh dài = long_side (giữ tỉ lệ) và nhãn [class, x1, y1, x2, y2] pixel trong ảnh đã co."""
        path = self.files[i]
        image = Image.open(path).convert("RGB")
        f = long_side / max(image.size)
        w, h = max(1, round(image.width * f)), max(1, round(image.height * f))
        image = np.array(image.resize((w, h), Image.BILINEAR))
        cls, cx, cy, bw, bh = torch.from_numpy(self.load_labels(path)).unbind(1)
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1) * torch.tensor([w, h, w, h])
        return image, torch.cat([cls[:, None], boxes], dim=1)

    def load_mosaic(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Mosaic của YOLOv8: 4 ảnh chạm nhau tại một điểm chung, nhìn qua một cửa sổ có zoom.

        Hình dung một mặt phẳng rộng 2s x 2s. Điểm chung J nằm ngẫu nhiên trong vùng giữa [s/2, 3s/2]. Ảnh i và 3
        ảnh ngẫu nhiên (cạnh dài s) mỗi ảnh nằm về một phía của J, một góc của ảnh chạm J:

                 ┌───────┐┌───────┐
                 │ (-,-) ││ (+,-) │        hướng (hx, hy) của từng ảnh so với J
                 └───────J└───────┘
                 ┌───────┐┌───────┐
                 │ (-,+) ││ (+,+) │
                 └───────┘└───────┘

        Ảnh đầu ra là cửa sổ cạnh s / zoom (zoom ngẫu nhiên 0.5..1.5) đặt quanh tâm mặt phẳng, lệch ngẫu nhiên ±10%,
        phóng về s x s. Thay vì dựng mặt phẳng rồi cắt, mỗi ảnh được co thẳng theo zoom và dán vào đầu ra:
        điểm P của mặt phẳng nằm ở (P - góc cửa sổ) x zoom trong ảnh đầu ra.
        """
        s = self.size
        rand = lambda lo, hi: lo + (hi - lo) * torch.rand(1).item()
        zoom = rand(0.5, 1.5)
        joint = torch.tensor([rand(0.5, 1.5), rand(0.5, 1.5)]) * s                # J, toạ độ mặt phẳng
        corner = torch.tensor([rand(0.9, 1.1), rand(0.9, 1.1)]) * s - s / (2 * zoom)  # góc trên-trái cửa sổ
        to_out = lambda p: (p - corner) * zoom                                     # mặt phẳng -> ảnh đầu ra
        plane = torch.cat([to_out(torch.zeros(2)), to_out(torch.full((2,), 2.0 * s))]).round()  # mép mặt phẳng
        frame = torch.tensor([0.0, 0.0, s, s])

        out = np.full((s, s, 3), 114, dtype=np.uint8)
        found = []
        for (hx, hy), j in zip([(-1, -1), (1, -1), (-1, 1), (1, 1)], [i] + torch.randint(len(self), (3,)).tolist()):
            img, lab = self.load_scaled(j, max(1, round(s * zoom)))
            size = torch.tensor([img.shape[1], img.shape[0]])
            j_out = to_out(joint).round()
            start = j_out - size * torch.tensor([hx == -1, hy == -1])              # góc trên-trái của ảnh ở đầu ra
            placed = torch.cat([start, start + size])
            ref = _intersect(placed, plane)                                        # phần ảnh nằm trên mặt phẳng
            vis = _intersect(ref, frame)                                           # ... và lọt vào cửa sổ
            l, t, r, b = vis.int().tolist()
            if r > l and b > t:
                sl, st = l - int(start[0]), t - int(start[1])
                out[t:b, l:r] = img[st:st + b - t, sl:sl + r - l]

            boxes = lab[:, 1:] + start.repeat(2)
            before, after = _intersect(boxes, ref), _intersect(boxes, vis)
            # giữ box còn thấy rõ: cạnh > 2 px và diện tích > 10% so với lúc chưa bị cửa sổ cắt
            seen, whole = _wh(after), _wh(before)
            keep = (seen > 2).all(1) & (seen.prod(1) > 0.1 * whole.prod(1).clamp(min=1e-6))
            found.append(torch.cat([lab[keep, :1], after[keep]], dim=1))
        return torch.from_numpy(out).permute(2, 0, 1).float() / 255, torch.cat(found)

    def __getitem__(self, i: int):
        if self.augment and torch.rand(1).item() < self.mosaic:
            x, labels = self.load_mosaic(i)
        else:
            path = self.files[i]
            image = np.array(Image.open(path).convert("RGB"))
            h, w = image.shape[:2]
            x, scale, (pad_x, pad_y) = letterbox(image, self.size)
            x = x[0]                                                          # (3, size, size)
            lab = torch.from_numpy(self.load_labels(path))                    # class, cx, cy, w, h (0..1)
            boxes = torch.stack([(lab[:, 1] - lab[:, 3] / 2) * w, (lab[:, 2] - lab[:, 4] / 2) * h,
                                 (lab[:, 1] + lab[:, 3] / 2) * w, (lab[:, 2] + lab[:, 4] / 2) * h], dim=1)
            boxes = boxes * scale + torch.tensor([pad_x, pad_y, pad_x, pad_y])  # pixel trong ảnh letterbox
            labels = torch.cat([lab[:, :1], boxes], dim=1)

        if self.augment:
            if torch.rand(1) < 0.5:                                       # lật ngang
                x = x.flip(-1)
                labels[:, [1, 3]] = self.size - labels[:, [3, 1]]
            x = (x * (0.75 + 0.5 * torch.rand(1))).clamp(0, 1)            # sáng / tối hơn tối đa 25%
        return x, labels


def _intersect(boxes: torch.Tensor, rect: torch.Tensor) -> torch.Tensor:
    """Phần giao của box (..., 4) [x1, y1, x2, y2] với hình chữ nhật rect (4,); rỗng thì x2 < x1 hoặc y2 < y1."""
    return torch.cat([boxes[..., :2].maximum(rect[:2]), boxes[..., 2:].minimum(rect[2:])], dim=-1)


def _wh(boxes: torch.Tensor) -> torch.Tensor:
    """Rộng, cao của box (n, 4); box rỗng ra 0."""
    return (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)


def collate(batch):
    """Gom batch: ảnh (B, 3, S, S); nhãn đệm cho đủ M = số box nhiều nhất trong batch.

    Trả về imgs, gt_labels (B, M, 1), gt_bboxes (B, M, 4), mask_gt (B, M, 1).
    """
    imgs, labels = zip(*batch)
    m = max(len(l) for l in labels)
    gt = torch.zeros(len(batch), m, 5)
    mask = torch.zeros(len(batch), m, 1)
    for i, l in enumerate(labels):
        gt[i, :len(l)] = l
        mask[i, :len(l)] = 1
    return torch.stack(imgs), gt[..., :1], gt[..., 1:], mask
