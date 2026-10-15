"""Nhận diện vật thể từ camera theo thời gian thực bằng model tự viết. Nhấn q hoặc Esc để thoát.

    python webcam.py                      # chỉ hiện người (class "person"), model COCO 80 class
    python webcam.py --all                # hiện mọi vật COCO
    python webcam.py --size 320           # nhanh hơn khoảng 3-4 lần, kém chính xác hơn với vật nhỏ / xa
    python webcam.py --weights runs/bien-bao/best.pt --all    # dùng model biển báo đã train
    python webcam.py --camera 1           # camera thứ hai

Mặc định dùng yolov8n_mine.pt (trọng số COCO gốc đã đổi sang định dạng của repo; tạo bằng train.load_pretrained).
"""
import argparse
import sys
import time

import cv2
import numpy as np
import torch

from yolo.dataset import COCO_NAMES
from yolo.model import YOLOv8n
from yolo.postprocess import postprocess
from yolo.preprocess import letterbox


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n_mine.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--size", type=int, default=640, help="cạnh ảnh đưa vào model; phải chia hết cho 32")
    ap.add_argument("--all", action="store_true", help="hiện mọi class thay vì chỉ person")
    ap.add_argument("--no-mirror", action="store_true", help="không lật ngang ảnh (mặc định lật như gương)")
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = YOLOv8n(ckpt["nc"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    names = ckpt.get("names") or COCO_NAMES
    show_all = args.all or ckpt["nc"] != 80  # model không phải COCO thì không có "person" để lọc

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        sys.exit(f"Không mở được camera {args.camera}. Thử --camera 1, hoặc kiểm tra quyền camera của Windows.")
    print("Đang chạy. Nhấn q hoặc Esc trong cửa sổ camera để thoát.")

    fps, last = 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Không đọc được khung hình từ camera.")
            break
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x, scale, (pad_x, pad_y) = letterbox(rgb, args.size)
        with torch.inference_mode():
            dets = postprocess(model(x), args.conf)
        if not show_all:
            dets = dets[dets[:, 5] == 0]  # class 0 = person
        h, w = frame.shape[:2]
        for x1, y1, x2, y2, score, cls in dets.tolist():
            x1, x2 = (np.clip((np.array([x1, x2]) - pad_x) / scale, 0, w - 1)).astype(int)
            y1, y2 = (np.clip((np.array([y1, y2]) - pad_y) / scale, 0, h - 1)).astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{names[int(cls)]} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 3, max(th, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
        last = now
        cv2.putText(frame, f"{fps:.1f} FPS  {len(dets)} vat", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        cv2.imshow("YOLOv8n tu viet - nhan q de thoat", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
