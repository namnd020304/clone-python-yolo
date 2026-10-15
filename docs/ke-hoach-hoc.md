# Plan học: tự dựng lại YOLOv8n từ số 0

Đây là căn cứ gốc của repo: học gì, theo thứ tự nào, biết mình đúng bằng cách nào. Tiến độ thực tế nằm ở
[ban-giao/trang-thai.md](ban-giao/trang-thai.md); file này chỉ đổi khi đổi lộ trình.

## 1. Mục tiêu

1. Viết lại YOLOv8n (bản nhỏ nhất, khoảng 3,16 triệu tham số) bằng PyTorch, từng khối một, đến khi model nhận ảnh
   `(1, 3, 640, 640)` và trả ra `(1, 84, 8400)` **giống bản gốc của Ultralytics về từng con số**, không chỉ về shape.
2. Tự viết toàn bộ quá trình train (dữ liệu, gán nhãn, loss, vòng lặp, đo mAP) và chứng minh model tự viết học được.
3. Dùng model thật: dataset riêng, ảnh, camera.
4. Khi đã học xong C++ C1–C3: viết lại bằng C++ / LibTorch.

## 2. Nguyên tắc học

- **Mỗi bước có bài chấm.** `python check.py N` chấm bước N. Viết xong bước nào chấm bước đó, qua rồi mới sang bước sau.
- **Tự viết trước, đọc Ultralytics sau.** Chỉ mở code của Ultralytics khi bước tương ứng đã qua bài chấm, để so cách viết.
- **Đúng shape chưa đủ, phải đúng số.** Các bước quan trọng được đối chiếu số với Ultralytics: đầu ra model (bước 7 +
  nạp trọng số), TAL (bước 12), loss (bước 13), mAP (bước 15).
- **Phép thử học thuộc trước, dữ liệu thật sau.** Model phải học thuộc được vài ảnh (mAP ≈ 0.995) thì loss và
  assigner mới đáng tin; sau đó mới train dataset thật.
- **Ba câu hỏi luôn đặt ra khi đọc code:** tensor này shape gì, đơn vị gì (pixel hay đơn vị ô)? Cái gì có gradient?
  Làm sao biết mình đúng?

## 3. Giai đoạn 1: dựng model (bước 1–9)

| Bước | File | Học / viết gì | Bài chấm kiểm tra |
|---|---|---|---|
| 1 | `yolo/blocks.py` | `Conv` = Conv2d + BatchNorm2d + SiLU | `(1,3,640,640)` → `(1,16,320,320)` với stride 2 |
| 2 | `yolo/blocks.py` | `Bottleneck`, `C2f` | giữ nguyên H, W; đổi số kênh |
| 3 | `yolo/blocks.py` | `SPPF` (3 lần MaxPool 5×5 nối lại) | giữ nguyên shape |
| 4 | `yolo/backbone.py` | Backbone 10 lớp, ra 3 mức P3, P4, P5 | `(64,80,80)`, `(128,40,40)`, `(256,20,20)` |
| 5 | `yolo/neck.py` | Neck PAN-FPN: trộn 3 mức lên rồi xuống | cùng 3 shape như bước 4 |
| 6 | `yolo/head.py` | Head tách nhánh box / class, DFL, anchor-free | `(1, 84, 8400)` |
| 7 | `yolo/model.py` | Ghép backbone + neck + head | ~3,2 triệu tham số |
| 8 | `yolo/preprocess.py` | Letterbox: resize giữ tỉ lệ + viền xám 114 | ảnh bất kỳ → `(1,3,640,640)` |
| 9 | `yolo/postprocess.py` | IoU, NMS, postprocess | NMS khớp `torchvision.ops.nms` |
| 9+ | `load_weights.py` | Nạp `yolov8n.pt` gốc vào model tự viết | đầu ra lệch Ultralytics ≤ 1e-4 |

Con số cần nhớ: `width = 0.25`, `depth = 0.33` → kênh 16/32/64/128/256, số lần lặp C2f 1/2/2/1; stride 8/16/32 →
lưới 80×80 + 40×40 + 20×20 = 8400 ô; mỗi ô 4 toạ độ + 80 class = 84.

## 4. Giai đoạn 2: dạy model tự học (bước 10–16)

Lý thuyết chi tiết từng bước: [train/](train/README.md).

| Bước | File | Học / viết gì | Kiểm chứng |
|---|---|---|---|
| 10 | `yolo/dataset.py` | Nhãn YOLO, đổi nhãn theo letterbox, gom batch có đệm + mask | `check.py 10`; vẽ nhãn lên ảnh, box khớp vật |
| 11 | `yolo/head.py` | Head trả đầu ra thô khi train (logit DFL, logit class, anchor, stride); khởi tạo bias | `check.py 11` |
| 12 | `yolo/assigner.py` | Task-Aligned Assigner: ô nào học vật nào | So với Ultralytics, `check.py 12` |
| 13 | `yolo/loss.py` | `7.5 × CIoU + 0.5 × BCE + 1.5 × DFL` | So với `v8DetectionLoss`, `check.py 13` |
| 14 | `train.py` | AdamW, warmup + cosine, cắt gradient, checkpoint | Học thuộc 4 ảnh coco8: mAP50-95 ≈ 0.995 |
| 15 | `yolo/metrics.py` | Precision, recall, AP, mAP50, mAP50-95 | So số `val` của Ultralytics trên `yolov8n.pt` |
| 16 | `yolo/dataset.py`, `yolo/ema.py` | Mosaic augmentation, EMA trọng số | `check.py 16`; vẽ ảnh mosaic kèm nhãn |

## 5. Giai đoạn 2b: dùng model thật

| Việc | File | Kiểm chứng |
|---|---|---|
| Lấy dataset từ Kaggle, đổi VOC → YOLO, chia train/val | `kaggle_data.py` | Kiểm tay vài box sau khi đổi |
| Fine-tune trên dataset riêng | `train.py --weights yolov8n.pt` | mAP trên tập val chưa từng thấy |
| Train dài trên máy ảo GCP (CPU, free trial) | xem `huong-dan/van-hanh.md` | Log, `best.pt` tải về, VM đã xoá |
| Dự đoán một ảnh | `predict.py` | Box đúng vật trên ảnh val |
| Nhận diện qua camera | `webcam.py` | Người chạy tự xem cửa sổ camera |

## 6. Phần còn lại của plan (chưa làm)

Theo thứ tự nên làm:

1. **Đo EMA + mosaic trên dataset thật:** train lại `bien-bao` rồi so với mAP50-95 0.705 (bản chưa có EMA + mosaic).
   Nên thêm **resume** (train tiếp từ `last.pt`) trước, để chạy máy ảo an toàn.
2. **Bổ sung augmentation còn thiếu so với Ultralytics:** đổi màu HSV, xoay / co giãn ngẫu nhiên.
3. **Tuỳ chọn:** AMP (số thực 16-bit) khi có GPU; mAP cho phép một box nhiều class giống cách Ultralytics đo.
4. **Giai đoạn 3: port sang C++ / LibTorch** (sau khi học xong C1–C3):
   - Xuất `best.pt` sang TorchScript để C++ đọc.
   - Viết lại letterbox, postprocess, NMS bằng C++; so kết quả với bản Python trên cùng ảnh.
   - Chạy camera bằng OpenCV C++.

## 7. Học song song C++

| Tuần C++ | Bước clone-yolo nên làm cùng lúc |
|---|---|
| C1 (biến, `for`, hàm, `vector`) | 1–3: tensor, shape, khối Conv / C2f / SPPF |
| C2 (`struct Box`, IoU, sort) | 4–7 và 9: ghép model, NMS (so với IoU tự viết bằng C++) |
| C3 (bộ nhớ, HWC → CHW) | 8: letterbox, `permute` HWC → CHW |
| Sau C3 | Giai đoạn 3: LibTorch |

## 8. Tài liệu tham khảo

- Kiến trúc: [Ultralytics YOLOv8 docs](https://docs.ultralytics.com/models/yolov8/), file cấu hình gốc
  `ultralytics/cfg/models/v8/yolov8.yaml` trên GitHub.
- PyTorch: [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html),
  [`torch.nn`](https://docs.pytorch.org/docs/stable/nn.html).
- Code Ultralytics để đối chiếu: `ultralytics/utils/tal.py` (TAL), `ultralytics/utils/loss.py` (`v8DetectionLoss`),
  `ultralytics/utils/torch_utils.py` (`ModelEMA`). Chỉ đọc sau khi bước tương ứng đã qua bài chấm.
