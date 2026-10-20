# Trạng thái

Cập nhật lần cuối: 17/10/2026. Cập nhật cuối mỗi buổi làm.

Cách đọc: danh sách bước và cách kiểm chứng nằm ở [../ke-hoach-hoc.md](../ke-hoach-hoc.md). File này chỉ ghi bước
đã xong, đang làm, bị chặn. Bước không có ở đây là chưa bắt đầu.

## Hiện tại

- Giai đoạn 1 (bước 1–9) và giai đoạn 2 (bước 10–16) đã viết xong. `check.py` qua 16/16 (12/10).
- Model tự viết khớp Ultralytics: đầu ra lệch 6,1e-05 khi nạp `yolov8n.pt`; loss sai khác 0,0.
- **Hướng mới (16/10, QĐ-10):** chỉ train từ số 0 trên COCO 2017, trên GPU máy dev (RTX 5070 Ti, ~14,5 phút/epoch).
  Không dùng biển báo và fine-tune nữa. Lệnh: [quy-trinh-train.md](quy-trinh-train.md) mục 3.2.
- Model biển báo cũ (lưu trữ, không phát triển tiếp): `runs/bien-bao/best.pt`, dataset biển báo 4 class, **mAP50-95 0.705, mAP50 0.836** trên
  175 ảnh val (12/10, 100 epoch trên máy ảo GCP, **chưa có** EMA + mosaic).
- Mọi thứ đã commit và push (`0fa7f9d`, 15/10). `best.pt` + `train.log` nằm ở
  [Release v1.0](https://github.com/namnd020304/clone-python-yolo/releases/tag/v1.0).
- **Đang chạy:** train YOLOv8n từ số 0 trên COCO 2017, 30 epoch, VM GCP `yolo-coco` (`us-east1-b`), bắt đầu
  05:06 UTC 15/10, dự kiến xong ~21/10. Xem: `gcloud compute ssh yolo-coco --zone=us-east1-b
  --command="tail -5 ~/yolo/runs/coco-scratch/train.log"`. Xong thì lấy `runs/` về và **xoá VM**.
- **Đã xong (17/10):** train COCO 30 epoch từ số 0 trên GPU máy dev (13:55 16/10 → 01:29 17/10, ~11,6 giờ): **mAP50-95 0.261, mAP50 0.354** (epoch 25; epoch 30 bằng 0.261). Bằng ~75% mốc 0.347 của `yolov8n.pt` gốc. Chưa hội tụ; chi tiết ở [nhat-ky-train.md](nhat-ky-train.md). Model: `runs/coco-gpu/best.pt`.
- Làm tiếp được ngay: so với job GCP khi xong; quyết có train dài hơn không (van-de-mo #8).

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
| — | Train COCO từ số 0 trên GCP CPU | Đang chạy (15/10) | 30 epoch, ~4,7 giờ/epoch |
| — | Train COCO từ số 0 trên GPU máy dev | Xong (17/10) | 30 epoch, ~11,6 giờ (epoch chậm gấp đôi từ epoch 16, chưa rõ lý do); mAP50-95 0.261 |
| — | Tài liệu `quy-trinh-train.md` | Xong (16/10) | Chỉ COCO, từ số 0 |
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
| Biển báo, EMA + mosaic | Bỏ, không đo (QĐ-10) | 16/10 |
| `yolov8n.pt` gốc trên val2017, code repo | mAP50 0.454 / mAP50-95 0.347 (Ultralytics công bố 0.373); 3,6 phút | 16/10 |
| Tốc độ train COCO, RTX 5070 Ti, batch 16 | 0 / 8 / 16 worker: 17 / 136 / 133 ảnh/s | 16/10 |
| **COCO từ số 0, 30 epoch, GPU máy dev** | **mAP50 0.354 / mAP50-95 0.261** (epoch 25; epoch 30: 0.353 / 0.261). Theo epoch 5/10/15/20/25/30: 0.137 / 0.213 / 0.235 / 0.247 / 0.261 / 0.261 | 17/10 |
| COCO từ số 0 so với `yolov8n.pt`, `bus.jpg` / `zidane.jpg`, conf 0.25 | `bus`: cùng 5 đối tượng, điểm thấp hơn 0,03–0,07. `zidane`: 2 person (0.90 / 0.62), thiếu tie; bản gốc 2 person 0.83 + tie 0.28. CPU 152 / 149 ms/ảnh. Chi tiết ở [nhat-ky-train.md](nhat-ky-train.md) | 18/10 |
| COCO từ số 0, 30 epoch, GCP CPU | Chưa có (chưa kiểm job) | |
| Camera CPU máy nhà, `--size` 640 / 480 / 320 | ~5 / ~6 / ~9,5 FPS | 12/10 |

Dữ liệu biển báo: Kaggle `andrewmvd/road-sign-detection`, 702 ảnh train, 175 ảnh val, 4 class
(crosswalk, speedlimit, stop, trafficlight).
