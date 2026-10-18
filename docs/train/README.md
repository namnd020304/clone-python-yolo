# Giai đoạn 2: dạy model tự học (train)

Ở giai đoạn 1 (bước 1–9) bạn đã dựng xong YOLOv8n: model nhận ảnh `(1, 3, 640, 640)` và trả ra `(1, 84, 8400)`.
Model đúng, nhưng trọng số **ngẫu nhiên** thì nó chẳng nhận ra gì. Muốn có model dùng được, ta phải *train*: cho model
xem ảnh có nhãn, đo xem nó sai bao nhiêu (loss), rồi chỉnh trọng số để lần sau bớt sai. Lặp lại hàng nghìn lần.

Bộ tài liệu này giải thích từng mảnh của quá trình đó, theo đúng thứ tự dữ liệu đi qua lúc train.

## Thứ tự học

| File | Học gì | Code | Chấm |
|---|---|---|---|
| [01-du-lieu.md](01-du-lieu.md) | Định dạng nhãn YOLO, đổi nhãn theo letterbox, augmentation (có mosaic), gom batch | [yolo/dataset.py](../../yolo/dataset.py) | `check.py 10`, `16` |
| [02-dau-ra-tho-cua-head.md](02-dau-ra-tho-cua-head.md) | Vì sao lúc train Head trả logit chứ không trả box; anchor, stride; khởi tạo bias | [yolo/head.py](../../yolo/head.py) | `check.py 11` |
| [03-gan-nhan-tal.md](03-gan-nhan-tal.md) | Bài toán "ô nào học vật nào"; Task-Aligned Assigner | [yolo/assigner.py](../../yolo/assigner.py) | `check.py 12` |
| [04-ham-loss.md](04-ham-loss.md) | BCE cho class, CIoU cho box, DFL cho phân bố khoảng cách | [yolo/loss.py](../../yolo/loss.py) | `check.py 13` |
| [05-vong-lap-train.md](05-vong-lap-train.md) | Gradient, AdamW, learning rate, warmup, EMA, train từ đầu (và vì sao không fine-tune) | [train.py](../../train.py), [yolo/ema.py](../../yolo/ema.py) | `check.py 14`, `16` |
| [06-danh-gia-map.md](06-danh-gia-map.md) | Precision, recall, AP, mAP50, mAP50-95 | [yolo/metrics.py](../../yolo/metrics.py) | `check.py 15` |
| [07-du-lieu-that-va-cloud.md](07-du-lieu-that-va-cloud.md) | COCO 2017, nhãn COCO JSON và YOLO, train trên GPU, vì sao cần worker | [yolo/dataset.py](../../yolo/dataset.py) | |

Nên đọc theo thứ tự: mỗi file dùng khái niệm của file trước. File 03 và 04 là phần khó nhất, cũng là phần quan trọng
nhất.

## Toàn cảnh một vòng lặp train

```
 ảnh + file nhãn .txt
        │  01: letterbox hoặc mosaic, đổi nhãn sang pixel 640x640, lật ngẫu nhiên, gom batch
        ▼
 imgs (B, 3, 640, 640)            gt_labels (B, M, 1), gt_bboxes (B, M, 4), mask_gt (B, M, 1)
        │                                              │
        ▼                                              │
 YOLOv8n ở chế độ train                                │
        │  02: Head trả đầu ra THÔ                      │
        ▼                                              │
 box (B, 64, 8400)  cls (B, nc, 8400)  anchors  strides │
        │                                              │
        ├──► decode DFL ra box dự đoán ──┐              │
        │                                ▼             ▼
        │                   03: TAL gán nhãn: ô nào học vật nào, nhãn class "mềm" bao nhiêu
        │                                │
        ▼                                ▼
 04: loss = 7.5 × (1 − CIoU) + 0.5 × BCE + 1.5 × DFL
        │
        ▼
 05: loss.backward() → gradient → AdamW chỉnh trọng số (lr theo warmup + cosine) → cập nhật EMA
        │
        └──► lặp lại; cứ vài epoch thì 06: đo mAP của bản EMA trên tập val, lưu best.pt
```

## Ba câu hỏi xuyên suốt

Khi đọc từng file, hãy luôn hỏi:

1. **Tensor này có shape gì, đơn vị gì?** Toạ độ lúc thì tính bằng pixel ảnh 640, lúc thì bằng "đơn vị ô" (pixel chia
   stride). Nhầm đơn vị là lỗi phổ biến nhất.
2. **Cái gì có gradient, cái gì không?** Phần gán nhãn (03) chạy trong `torch.no_grad()`: nó chỉ quyết định *học cái
   gì*, không phải thứ được học.
3. **Làm sao biết mình đúng?** Mỗi bước có một bài trong `check.py`. Bước 12 và 13 còn so con số với Ultralytics;
   loss của repo này khớp Ultralytics tới sai khác 0.0.

## Kết quả đã đạt được

| Thí nghiệm | Kết quả |
|---|---|
| Đo `yolov8n.pt` gốc trên coco8 val | mAP50-95 0.600 (Ultralytics đo được 0.608) |
| Fine-tune 30 epoch trên coco8, hướng cũ | mAP50-95 val 0.600 → 0.636 ở epoch 10 |
| Train từ số 0, 300 epoch, 4 ảnh | Học thuộc: mAP50-95 0.995 trên chính 4 ảnh đó |
| Fine-tune 3 epoch, dataset biển báo (702 ảnh, 4 class), hướng cũ | mAP50 0.024 → 0.716, mAP50-95 0.019 → 0.581 |
| Đo `yolov8n.pt` gốc trên val2017 (5.000 ảnh) | mAP50 0.454, mAP50-95 0.347 (Ultralytics công bố 0.373) |
| Train từ số 0 trên COCO 2017 | Chưa có (hướng hiện tại, QĐ-10) |
