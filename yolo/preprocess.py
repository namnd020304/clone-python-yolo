"""Bước 8: chuẩn bị ảnh cho model."""
import numpy as np
import torch


def letterbox(image: np.ndarray, size: int = 640, pad_value: int = 114):
    """Resize ảnh HWC uint8 (RGB) về cạnh dài = size, giữ tỉ lệ; thêm viền màu pad_value để thành size x size,
    ảnh nằm giữa.

    Trả về (tensor, scale, (pad_x, pad_y)):
      - tensor: float32 (1, 3, size, size), giá trị 0..1, thứ tự CHW.
      - scale: tỉ lệ đã nhân vào ảnh gốc.
      - pad_x, pad_y: số pixel viền bên trái và bên trên (dùng để đổi box về toạ độ ảnh gốc).

    Gợi ý: torch.nn.functional.interpolate để resize; permute(2, 0, 1) để đổi HWC -> CHW.
    """
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255
    x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out = torch.full((1, 3, size, size), pad_value / 255, dtype=torch.float32)
    out[:, :, pad_y:pad_y + nh, pad_x:pad_x + nw] = x
    return out, scale, (pad_x, pad_y)
