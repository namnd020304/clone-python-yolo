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
      - mosaic với xác suất self.mosaic (mặc định 1.0): ghép 4 ảnh, co giãn ngẫu nhiên x0.5..1.5 rồi cắt về size.
        train.py tắt mosaic (đặt self.mosaic = 0) ở vài epoch cuối, giống Ultralytics.
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

    def load_resized(self, i: int) -> tuple[np.ndarray, torch.Tensor]:
        """Ảnh thứ i co về cạnh dài = size (giữ tỉ lệ, chưa thêm viền) và nhãn [class, x1, y1, x2, y2] pixel của ảnh đó."""
        path = self.files[i]
        image = Image.open(path).convert("RGB")
        r = self.size / max(image.size)
        w, h = max(1, round(image.width * r)), max(1, round(image.height * r))
        image = np.array(image.resize((w, h), Image.BILINEAR))
        lab = torch.from_numpy(self.load_labels(path))
        boxes = torch.stack([(lab[:, 1] - lab[:, 3] / 2) * w, (lab[:, 2] - lab[:, 4] / 2) * h,
                             (lab[:, 1] + lab[:, 3] / 2) * w, (lab[:, 2] + lab[:, 4] / 2) * h], dim=1)
        return image, torch.cat([lab[:, :1], boxes], dim=1)

    def load_mosaic(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Ghép ảnh i và 3 ảnh ngẫu nhiên vào khung 2s x 2s quanh tâm (xc, yc) ngẫu nhiên:

            ┌─────────┬─────────┐
            │  ảnh 0  │  ảnh 1  │      góc dưới-phải của ảnh 0, góc dưới-trái của ảnh 1, ...
            ├─────────●─────────┤      cùng chạm vào tâm ● = (xc, yc)
            │  ảnh 2  │  ảnh 3  │
            └─────────┴─────────┘

        Rồi lấy một cửa sổ cỡ s / r (r ngẫu nhiên 0.5..1.5) quanh tâm khung, dịch ngẫu nhiên ±10%, co về s x s.
        r < 1: cửa sổ lớn, vật bị thu nhỏ (luyện vật nhỏ); r > 1: vật được phóng to.
        """
        s = self.size
        pad = s // 2                                       # lề quanh khung để cửa sổ có tràn ra cũng không lỗi
        canvas = np.full((2 * s + 2 * pad, 2 * s + 2 * pad, 3), 114, dtype=np.uint8)
        xc, yc = (int(s * (0.5 + torch.rand(1).item())) for _ in range(2))  # tâm trong [s/2, 3s/2]
        ids = [i] + torch.randint(len(self), (3,)).tolist()
        labels = []
        for k, j in enumerate(ids):
            img, lab = self.load_resized(j)
            h, w = img.shape[:2]
            if k == 0:    # trên-trái: góc dưới-phải chạm tâm
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            elif k == 1:  # trên-phải
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
            elif k == 2:  # dưới-trái
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, 2 * s)
            else:         # dưới-phải
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(yc + h, 2 * s)
            # phần ảnh nguồn được dán: ảnh bên trái/trên bị cắt mất phần xa tâm
            x1b = w - (x2a - x1a) if k in (0, 2) else 0
            y1b = h - (y2a - y1a) if k in (0, 1) else 0
            canvas[pad + y1a:pad + y2a, pad + x1a:pad + x2a] = img[y1b:y1b + (y2a - y1a), x1b:x1b + (x2a - x1a)]
            lab[:, [1, 3]] += x1a - x1b                    # toạ độ trong ảnh nguồn -> toạ độ trong khung 2s
            lab[:, [2, 4]] += y1a - y1b
            lab[:, 1:] = torch.max(torch.min(lab[:, 1:], torch.tensor([x2a, y2a, x2a, y2a])),
                                   torch.tensor([x1a, y1a, x1a, y1a]))  # cắt box theo phần ảnh thực sự được dán
            labels.append(lab)
        labels = torch.cat(labels)

        # cửa sổ s / r quanh tâm khung (s, s), dịch ±0.1 s
        r = 0.5 + torch.rand(1).item()
        win = s / r
        cx, cy = (s + (torch.rand(1).item() * 0.2 - 0.1) * s for _ in range(2))
        x0, y0 = cx - win / 2, cy - win / 2
        crop = Image.fromarray(canvas).resize((s, s), Image.BILINEAR,
                                              box=(x0 + pad, y0 + pad, x0 + pad + win, y0 + pad + win))
        x = torch.from_numpy(np.array(crop)).permute(2, 0, 1).float() / 255

        before = labels[:, 1:].clone()                     # box trong khung, chưa cắt theo cửa sổ
        boxes = (before - torch.tensor([x0, y0, x0, y0])) * r
        boxes = boxes.clamp(0, s)
        # bỏ box bị cắt gần hết: cạnh < 2 px, hoặc còn < 10% diện tích (so với lúc chưa cắt)
        wh = boxes[:, 2:] - boxes[:, :2]
        wh0 = (before[:, 2:] - before[:, :2]) * r
        keep = (wh > 2).all(1) & (wh.prod(1) > 0.1 * wh0.prod(1).clamp(min=1e-6))
        return x, torch.cat([labels[keep, :1], boxes[keep]], dim=1)

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
