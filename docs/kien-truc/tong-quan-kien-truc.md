# Tổng quan kiến trúc

File này nói plan đã thành code thế nào. Lộ trình và lý do học theo thứ tự này nằm trong
[../ke-hoach-hoc.md](../ke-hoach-hoc.md); lý thuyết train nằm trong [../train/](../train/README.md); các quyết định
đã chốt nằm trong [quyet-dinh.md](quyet-dinh.md).

## 1. Ranh giới

```
 ../data/coco (COCO 2017) / datasets/coco8 ──► train.py (từ số 0) ──► runs/<tên>/best.pt ──► predict.py / webcam.py

 yolov8n.pt (Ultralytics) ──► load_weights.py, check.py (chỉ đối chiếu, làm mốc mAP; không dùng làm điểm xuất phát)
```

Gói `yolo/` là model và mọi thứ cần để train, **không phụ thuộc Ultralytics**. Ultralytics chỉ được dùng ở ba chỗ, đều
để đối chiếu hoặc lấy trọng số: `load_weights.py`, `check.py` (bài 12, 13) và `train.load_pretrained` khi đọc file
`yolov8n.pt` gốc. `predict.py` và `webcam.py` chạy được không cần Ultralytics.

## 2. Bản đồ module

| File | Trách nhiệm | Bước | Trạng thái |
|---|---|---|---|
| `yolo/blocks.py` | `Conv`, `Bottleneck`, `C2f`, `SPPF` | 1–3 | Xong |
| `yolo/backbone.py` | 10 lớp, ra P3 / P4 / P5 | 4 | Xong |
| `yolo/neck.py` | PAN-FPN: lên (upsample + concat + C2f) rồi xuống (Conv stride 2 + concat + C2f) | 5 | Xong |
| `yolo/head.py` | `make_anchors`, `DFL`, `Head`: nhánh box 64 kênh + nhánh class nc kênh cho mỗi mức; `init_bias` | 6, 11 | Xong |
| `yolo/model.py` | `YOLOv8n(num_classes)` = backbone → neck → head | 7 | Xong |
| `yolo/preprocess.py` | `letterbox` → tensor, `scale`, `(pad_x, pad_y)` | 8 | Xong |
| `yolo/postprocess.py` | `box_iou`, `bbox_ciou`, `nms`, `postprocess` | 9, 13 | Xong |
| `yolo/dataset.py` | `YoloDataset` (letterbox / mosaic, lật, độ sáng), `collate`, `download_coco8` | 10, 16 | Xong |
| `yolo/assigner.py` | `TaskAlignedAssigner` | 12 | Xong |
| `yolo/loss.py` | `DetectionLoss`, `dfl_loss` | 13 | Xong |
| `yolo/metrics.py` | `match_predictions`, `average_precision`, `mean_ap`, `evaluate` | 15 | Xong |
| `yolo/ema.py` | `ModelEMA` | 16 | Xong (mosaic + EMA sẽ đánh giá qua train COCO) |
| `train.py` | Vòng lặp train, `load_pretrained`, optimizer, lịch lr | 14 | Xong |
| `check.py` | 16 bài chấm | — | Xong |
| `load_weights.py` | Nạp `yolov8n.pt`, so đầu ra với Ultralytics, vẽ `result.jpg` | 9+ | Xong |
| `kaggle_data.py` | Tải và chuyển dataset (YOLO / VOC) về cấu trúc chuẩn | 2b | Xong, không dùng nữa (QĐ-10) |
| `predict.py` | Chạy checkpoint trên một ảnh | 2b | Xong |
| `webcam.py` | Chạy theo thời gian thực từ camera | 2b | Code xong, chưa có người xem chạy thật |

## 3. Luồng chính

### Suy luận (một ảnh)

```
ảnh HWC uint8 ─► letterbox (giữ tỉ lệ, viền 114) ─► (1, 3, 640, 640) float 0..1
 ─► Backbone ─► P3 (64,80,80)  P4 (128,40,40)  P5 (256,20,20)
 ─► Neck     ─► N3, N4, N5 cùng shape
 ─► Head (eval): box 64 kênh/ô ─► DFL (softmax 16 bin, kỳ vọng) ─► l,t,r,b (đơn vị ô) ─► cx,cy,w,h (pixel)
                 class nc kênh/ô ─► sigmoid
 ─► (1, 4 + nc, 8400)
 ─► postprocess: lọc theo conf ─► xyxy ─► NMS theo class (cộng class_id × 4096 vào toạ độ) ─► [x1,y1,x2,y2,score,cls]
 ─► trừ pad, chia scale ─► toạ độ ảnh gốc
```

### Train (một bước)

```
datasets/<tên>/images/train/*.jpg + labels/train/*.txt (class cx cy w h, 0..1)
 ─► YoloDataset: mosaic (xác suất --mosaic, tắt ở --close-mosaic epoch cuối) hoặc letterbox; lật ngang; độ sáng
 ─► collate: imgs (B,3,640,640); gt_labels (B,M,1), gt_bboxes (B,M,4) pixel xyxy, mask_gt (B,M,1) (đệm tới M box)
 ─► Head (train): box (B,64,8400) logit, cls (B,nc,8400) logit, anchors (8400,2), strides (8400,1)
 ─► decode DFL ra box dự đoán ─► TAL (no_grad): top-10 ô theo score^0.5 × CIoU^6, tâm ô nằm trong gt; nhãn class mềm
 ─► loss (3,) = [7.5 × (1 − CIoU), 0.5 × BCE, 1.5 × DFL], mỗi phần chia cho tổng nhãn mềm (≈ số vật)
 ─► backward ─► clip_grad_norm 10 ─► AdamW (lr warmup tuyến tính rồi cosine về 1%) ─► cập nhật EMA
 ─► mỗi --eval-every epoch: evaluate(EMA) ra mAP50, mAP50-95 ─► lưu last.pt; tốt hơn thì lưu best.pt
```

Hai đơn vị toạ độ cùng tồn tại: **pixel ảnh 640** (gt, box ra cuối) và **đơn vị ô** (pixel chia stride, dùng trong DFL
và anchor). Nhầm hai đơn vị là lỗi hay gặp nhất.

## 4. Nơi lưu

| Thứ | Ở đâu | Trong git? |
|---|---|---|
| Code, tài liệu | gốc repo, `yolo/`, `docs/` | Có |
| Trọng số gốc | `yolov8n.pt`, `yolov8n_mine.pt` ở gốc repo | Không (`*.pt`) |
| Dataset | `datasets/<tên>/` (+ `classes.txt`) | Không |
| Kết quả train | `runs/<tên>/last.pt`, `best.pt`, `train.log`, ảnh `pred_*.jpg` | Không |
| Ảnh đầu ra | `result.jpg` (load_weights), `predict.jpg` (predict) | Không |
| Môi trường | `venv/` | Không |

Checkpoint của `train.py` là một dict `{"model", "epoch", "nc", "names", "map"}`; `model` là state_dict của bản EMA
(hoặc của model nếu `--no-ema`). `predict.py` và `webcam.py` dựng `YOLOv8n(ckpt["nc"])` rồi nạp `ckpt["model"]`.
File `yolov8n.pt` của Ultralytics có định dạng khác (lưu cả `nn.Module`), nên phải qua `load_weights.py` hoặc
`train.load_pretrained`.
