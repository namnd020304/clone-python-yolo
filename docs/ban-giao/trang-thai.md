# Trạng thái

Cập nhật lần cuối: 15/10/2026. Cập nhật cuối mỗi buổi làm.

Cách đọc: danh sách bước và cách kiểm chứng nằm ở [../ke-hoach-hoc.md](../ke-hoach-hoc.md). File này chỉ ghi bước
đã xong, đang làm, bị chặn. Bước không có ở đây là chưa bắt đầu.

## Hiện tại

- Giai đoạn 1 (bước 1–9) và giai đoạn 2 (bước 10–16) đã viết xong. `check.py` qua 16/16 (12/10).
- Model tự viết khớp Ultralytics: đầu ra lệch 6,1e-05 khi nạp `yolov8n.pt`; loss sai khác 0,0.
- Model tốt nhất hiện có: `runs/bien-bao/best.pt`, dataset biển báo 4 class, **mAP50-95 0.705, mAP50 0.836** trên
  175 ảnh val (12/10, 100 epoch trên máy ảo GCP, **chưa có** EMA + mosaic).
- Mọi thứ đã commit và push (`0fa7f9d`, 15/10). `best.pt` + `train.log` nằm ở
  [Release v1.0](https://github.com/namnd020304/clone-python-yolo/releases/tag/v1.0).
- **Đang chạy:** train YOLOv8n từ số 0 trên COCO 2017, 30 epoch, VM GCP `yolo-coco` (`us-east1-b`), bắt đầu
  05:06 UTC 15/10, dự kiến xong ~21/10. Xem: `gcloud compute ssh yolo-coco --zone=us-east1-b
  --command="tail -5 ~/yolo/runs/coco-scratch/train.log"`. Xong thì lấy `runs/` về và **xoá VM**.
- Làm tiếp được ngay: train lại `bien-bao` có EMA + mosaic để so với 0.705 (van-de-mo #1).

## Theo plan

| Bước | Việc | Trạng thái | Ghi chú |
|---|---|---|---|
| 1–3 | `Conv`, `Bottleneck`, `C2f`, `SPPF` | Xong (09/10) | Sửa 11/10: `SPPF` padding `k // 2`; BatchNorm `eps=1e-3, momentum=0.03` (QĐ-03) |
| 4–7 | Backbone, Neck, Head + DFL, ghép model | Xong (09/10) | 3.157.200 tham số, khớp bản gốc |
| 8 | Letterbox | Xong (09/10) | |
| 9 | IoU, NMS, postprocess | Xong (09/10) | NMS khớp `torchvision.ops.nms` |
| 9+ | Nạp `yolov8n.pt` gốc (`load_weights.py`) | Xong (11/10) | 354 tensor nạp vừa khít; lệch 6,1e-05; `bus.jpg` ra 4 người + 1 xe buýt |
| 10 | Dataset + collate | Xong (11/10) | Tự tải coco8; đọc `classes.txt` |
| 11 | Head trả đầu ra thô khi train, khởi tạo bias | Xong (11/10) | |
| 12 | TAL assigner | Xong (11/10) | Khớp Ultralytics |
| 13 | Loss | Xong (11/10) | Sai khác 0,0 ở box, cls, dfl |
| 14 | `train.py` | Xong (11/10) | Học thuộc 4 ảnh coco8 từ số 0: mAP50-95 0.995 ở epoch 200 |
| 15 | mAP | Xong (11/10) | `yolov8n.pt` trên coco8 val: 0.600 (Ultralytics 0.608) |
| 16 | Mosaic + EMA | Code xong, **chưa đo trên dữ liệu thật** (12/10) | Học thuộc với EMA vẫn đạt 0.995. Cần train lại `bien-bao` |
| 2b | `kaggle_data.py` | Xong (11/10) | Hỗ trợ YOLO `.txt` và Pascal VOC; chưa hỗ trợ COCO JSON, CSV, đa giác |
| 2b | Fine-tune biển báo trên GCP | Xong (12/10) | 100 epoch, ~100 s/epoch, VM đã xoá |
| 2b | `predict.py` | Xong (11/10) | |
| 2b | `webcam.py` | Code xong, **chưa có người xem cửa sổ camera chạy thật** (12/10) | Đã kiểm phần xử lý trên `bus.jpg` và đọc được khung hình |
| 2b | Báo cáo `docs/bao-cao-yolov8n.docx` | Xong (12/10) | Chưa xem giao diện trong Word |
| — | Tài liệu bàn giao `docs/` | Xong bản đầu (15/10) | |
| 6.1 | Resume từ `last.pt` | Xong (15/10) | `--resume`; thử trên coco8 |
| — | Train COCO từ số 0 | Đang chạy (15/10) | 30 epoch, ~4,7 giờ/epoch |
| 6.2 | HSV, xoay / co giãn | Chưa làm | |
| 3 | Port C++ / LibTorch | Chưa làm | Chờ học xong C1–C3 |

## Kết quả đo

| Thí nghiệm | Kết quả | Ngày |
|---|---|---|
| Đầu ra model tự viết so với Ultralytics (nạp `yolov8n.pt`) | lệch lớn nhất 6,1e-05 | 11/10 |
| Loss so với `v8DetectionLoss` cùng batch | sai khác 0,0 (box, cls, dfl) | 11/10 |
| `yolov8n.pt` trên coco8 val | mAP50 0.758 / mAP50-95 0.600 (Ultralytics 0.808 / 0.608) | 11/10 |
| Fine-tune coco8, 30 epoch | mAP50-95 0.600 → 0.636 ở epoch 10, sau đó giảm (học thuộc 4 ảnh) | 11/10 |
| Train từ số 0, 4 ảnh coco8, 300 epoch | mAP50-95 0.995 (không EMA) / 0.995 (có EMA) | 11/10, 12/10 |
| Biển báo, fine-tune 3 epoch (máy nhà) | mAP50 0.024 → 0.716, mAP50-95 0.019 → 0.581 | 11/10 |
| **Biển báo, fine-tune 100 epoch (GCP)** | **best epoch 75: mAP50 0.836, mAP50-95 0.705** | 12/10 |
| Biển báo, EMA + mosaic | Chưa đo | |
| Camera CPU máy nhà, `--size` 640 / 480 / 320 | ~5 / ~6 / ~9,5 FPS | 12/10 |

Dữ liệu biển báo: Kaggle `andrewmvd/road-sign-detection`, 702 ảnh train, 175 ảnh val, 4 class
(crosswalk, speedlimit, stop, trafficlight).
