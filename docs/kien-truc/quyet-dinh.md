# Nhật ký quyết định

Mỗi quyết định có mã QĐ-NN. Không xoá quyết định cũ. Khi đổi, thêm quyết định mới và sửa trạng thái quyết định cũ
thành "Thay bằng QĐ-xx". Đọc file này để biết vì sao code như hiện tại và khi nào nên xem lại.

Mẫu cho quyết định mới:

```
## QĐ-NN. <Tên>
- Ngày: dd/mm/yyyy · Trạng thái: Đang áp dụng
- Bối cảnh: …
- Quyết định: …
- Lý do: …
- Phương án đã loại: …
- Xem lại khi: …
```

## QĐ-01. PyTorch thuần, tự viết mọi khối; Ultralytics chỉ để đối chiếu

- Ngày: 09/10/2026 · Trạng thái: Đang áp dụng
- Quyết định: gói `yolo/` chỉ dùng `torch`, `numpy`, `Pillow`. Không import Ultralytics trong `yolo/`. Ultralytics chỉ
  dùng để tải `yolov8n.pt` và so số (`load_weights.py`, `check.py` bài 12–13).
- Lý do: mục tiêu là học; dùng lại code Ultralytics thì không học được gì. Đối chiếu số với Ultralytics là cách duy
  nhất biết mình đúng thật chứ không chỉ đúng shape.
- Xem lại khi: port sang LibTorch (giai đoạn 3) — lúc đó bản Python của repo này trở thành bản đối chiếu.

## QĐ-02. Mỗi bước một bài chấm trong `check.py`

- Ngày: 09/10/2026 · Trạng thái: Đang áp dụng
- Quyết định: mỗi bước của plan có một hàm `stepN` trong `check.py`; bước chưa viết (`NotImplementedError`) báo
  "chưa làm". Bước mới thì thêm bài chấm trước hoặc cùng lúc với code.
- Lý do: học từng bước nhỏ, biết ngay sai ở đâu.

## QĐ-03. BatchNorm `eps=1e-3, momentum=0.03` như Ultralytics

- Ngày: 11/10/2026 · Trạng thái: Đang áp dụng
- Bối cảnh: mặc định của PyTorch là `eps=1e-5, momentum=0.1`.
- Quyết định: `nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)` trong `Conv`.
- Lý do: đo thật — với `eps=1e-5` đầu ra lệch Ultralytics tới 141 khi nạp `yolov8n.pt`; với `1e-3` chỉ lệch 6,1e-05.
  `eps` nằm trong công thức chuẩn hoá nên trọng số train với giá trị nào phải chạy với đúng giá trị đó.
- Xem lại khi: nạp trọng số từ nguồn khác Ultralytics.

## QĐ-04. Nạp `yolov8n.pt` bằng bảng đổi tên khoá

- Ngày: 11/10/2026 · Trạng thái: Đang áp dụng
- Quyết định: `load_weights.py` đổi tên từng khoá của model tự viết sang khoá Ultralytics (`LAYERS`: `backbone.c2f1` →
  `model.2`…; `PARTS`: `conv1` → `cv1`, `bottlenecks` → `m`). Khoá thiếu, thừa hoặc khác shape thì báo lỗi ngay.
  Bỏ qua `head.dfl.project` (buffer 0..15 tự tạo; bản gốc lưu dạng conv weight `model.22.dfl.conv.weight`).
- Lý do: đổi tên tường minh bắt được mọi lệch kiến trúc, thay vì ghép theo thứ tự và âm thầm nạp sai.
- Hệ quả: đổi tên thuộc tính trong `yolo/` thì phải sửa bảng.

## QĐ-05. Head trả đầu ra thô khi `model.train()`, đã decode khi `model.eval()`

- Ngày: 11/10/2026 · Trạng thái: Đang áp dụng
- Quyết định: chế độ train trả `(box logit (B,64,A), cls logit (B,nc,A), anchors, strides)`; chế độ eval trả
  `(B, 4 + nc, A)` như cũ.
- Lý do: loss cần logit (BCE ổn định số hơn khi tự làm sigmoid bên trong; DFL loss cần phân bố 16 bin). Đầu ra lúc dự
  đoán không đổi nên bước 1–9 và `load_weights.py` vẫn đúng.
- Hệ quả: nhớ gọi `model.eval()` trước khi dự đoán hoặc đo mAP.

## QĐ-06. Learning rate mặc định `0.002 × 5 / (4 + nc)`

- Ngày: 11/10/2026 · Trạng thái: Đang áp dụng
- Bối cảnh: lr cố định 0.002 phá hỏng trọng số gốc sau vài bước khi fine-tune 80 class (mAP về 0 dù loss vẫn giảm).
- Quyết định: `--lr 0` (mặc định) tự tính như Ultralytics: ≈ 0.00012 với 80 class, ≈ 0.0013 với 4 class. AdamW, weight
  decay chỉ cho trọng số conv; warmup tuyến tính rồi cosine về 1%; cắt gradient norm 10.
- Xem lại khi: train từ số 0 (phép thử học thuộc dùng `--lr 0.002`).

## QĐ-07. Fine-tune: bỏ qua tensor khác shape

- Ngày: 11/10/2026 · Trạng thái: Đang áp dụng
- Quyết định: `train.load_pretrained` nạp mọi tensor trùng tên và shape, bỏ qua phần còn lại (lớp class cuối khi số
  class khác 80). Với 4 class: nạp 319/355 tensor.
- Lý do: backbone, neck, nhánh box vẫn dùng được; chỉ nhánh class phải học lại. Ultralytics làm tương tự.

## QĐ-08. Train dài trên máy ảo GCP chỉ CPU

- Ngày: 11/10/2026 · Trạng thái: Thay bằng QĐ-10 (job COCO chạy từ 15/10 vẫn để chạy hết)
- Bối cảnh: CPU máy nhà ~2,5–3 phút/epoch với 702 ảnh; tài khoản GCP là free trial, không có hạn mức GPU.
- Quyết định: VM `n2-standard-8` (8 vCPU), `us-central1-a`; gửi code (không gửi `venv`, `datasets`, `runs`), tải dataset
  ngay trên VM, chạy trong `tmux`, VM tự tắt khi train xong. Lấy kết quả xong thì **xoá VM** (ổ đĩa vẫn tính tiền khi
  VM tắt).
- Kết quả: ~100 s/epoch, 100 epoch ~2 giờ 50 phút, ~1,2 USD credit (ước tính).
- Phương án đã loại: VM GPU T4/L4 spot (nhanh hơn nhiều, rẻ) — cần tài khoản trả phí để xin hạn mức GPU.
- Xem lại khi: có tài khoản trả phí, hoặc dùng Google Colab có GPU.

## QĐ-09. Mosaic + EMA bật mặc định

- Ngày: 12/10/2026 · Trạng thái: Đang áp dụng, **chưa đo trên dữ liệu thật** (van-de-mo #1)
- Quyết định: `--mosaic 1.0`, tắt mosaic ở `--close-mosaic 10` epoch cuối; EMA `d = 0.9999 × (1 − e^(−bước/2000))`, đo
  mAP và lưu checkpoint bằng bản EMA. Tắt bằng `--no-augment` / `--no-ema`.
- Lý do: mosaic thu nhỏ vật trong ảnh ghép, nhắm đúng điểm yếu vật nhỏ; EMA cho mAP ổn định hơn. Cả hai là mặc định
  của Ultralytics.
- Xem lại khi: có số đo trên `bien-bao`. Nếu không tăng mAP thì cân nhắc tắt mặc định với dataset nhỏ. Từ 16/10 không
  dùng `bien-bao` nữa (QĐ-10): đánh giá lại bằng kết quả train COCO.

## QĐ-10. Chỉ train từ số 0 trên COCO 2017, trên GPU máy dev

- Ngày: 16/10/2026 · Trạng thái: Đang áp dụng
- Bối cảnh: mục tiêu là tự build model, không phải tinh chỉnh model người khác. Fine-tune từ `yolov8n.pt` (trên biển
  báo hay COCO) chỉ chứng minh code nạp và chỉnh được trọng số có sẵn. Máy dev có RTX 5070 Ti 16 GB: ~14,5 phút/epoch
  COCO, so với ~4,7 giờ/epoch trên VM GCP chỉ CPU.
- Quyết định: chỉ train **từ số 0** (không `--weights`) trên COCO 2017 (`train2017` / `val2017`, 80 class), chạy trên
  GPU máy dev. Bỏ dataset biển báo và fine-tune. `yolov8n.pt` chỉ còn dùng để đối chiếu (`load_weights.py`, `check.py`)
  và làm mốc mAP (0.347 trên val2017 khi đo bằng code repo). Không mở thêm job COCO trên GCP CPU.
- Lệnh chuẩn: xem [../ban-giao/quy-trinh-train.md](../ban-giao/quy-trinh-train.md) mục 3.2.
- Xem lại khi: kết quả 30 epoch từ số 0 quá thấp, cần quyết train dài hơn hay đổi optimizer / lr.
