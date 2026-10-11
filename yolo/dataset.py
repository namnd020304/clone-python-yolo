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
    """Trả về (ảnh (3, size, size) float 0..1, nhãn (n, 5) = [class, x1, y1, x2, y2] pixel trong ảnh đã letterbox).

    augment=True: lật ngang ngẫu nhiên và đổi độ sáng nhẹ. Đủ cho thí nghiệm nhỏ; Ultralytics còn dùng mosaic,
    xoay, co giãn, đổi HSV.
    """

    def __init__(self, img_dir: str | Path, size: int = 640, augment: bool = False):
        self.files = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
        if not self.files:
            raise FileNotFoundError(f"không có ảnh trong {img_dir}")
        self.size = size
        self.augment = augment

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

    def __getitem__(self, i: int):
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
