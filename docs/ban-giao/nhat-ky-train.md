# Nhật ký quá trình train

Ghi riêng những gì **phát sinh trong lúc train**: cấu hình từng lần chạy, số đo theo epoch, sự cố, việc đã xử lý.
Nhật ký công việc chung (viết code, đổi hướng) ở [nhat-ky.md](nhat-ky.md); cách chạy train ở
[quy-trinh-train.md](quy-trinh-train.md).

Quy ước: mỗi lần chạy một mục `## Lần N`, lần mới nhất ở trên. Trong mục có: cấu hình, bảng theo epoch, sự cố, kết
luận. Số lấy từ `train.log`, không nhớ lại. Chưa có thì ghi "chưa có", không đoán.

## Tổng hợp các lần chạy

| Lần | Dữ liệu | Máy | Cấu hình chính | Trạng thái | mAP50-95 tốt nhất |
|---|---|---|---|---|---|
| 3 | COCO 2017, từ số 0 | GPU máy dev (RTX 5070 Ti) | 30 epoch, batch 16, 8 worker, lr 0.001, đo mAP mỗi 5 epoch | **Xong** 01:29 17/10, ~11,6 giờ | **0.261** (epoch 25 và 30) |
| 2 | COCO 2017, từ số 0 | VM GCP `yolo-coco` (CPU, `us-east1-b`) | 30 epoch, lr 0.001, đo mAP mỗi epoch | Đang chạy từ 05:06 UTC 15/10, **chưa kiểm lại** từ 16/10 (máy dev chưa có `gcloud`) | Chưa có |
| 1 | Biển báo 4 class, fine-tune từ `yolov8n.pt` | VM GCP (CPU) | 100 epoch, ~100 s/epoch | Xong 12/10, VM đã xoá. Lưu trữ, không phát triển tiếp (QĐ-10) | 0.705 (epoch 75) |

Ngoài ra: `yolov8n.pt` gốc đo bằng code repo trên val2017 được mAP50-95 0.347 (mốc so sánh; Ultralytics công bố 0.373,
xem van-de-mo #9).

---

## Lần 3: COCO từ số 0 trên GPU máy dev (xong 17/10)

**Cấu hình**

```powershell
python -u train.py --data ../data/coco --train-split train2017 --val-split val2017 --epochs 30 --batch 16 `
    --workers 8 --lr 0.001 --eval-every 5 --out runs/coco-gpu > runs/coco-gpu.log
```

- Bắt đầu khoảng 13:55 ngày 16/10/2026 (theo giờ tạo thư mục `runs/coco-gpu/`). Dữ liệu: 118.287 ảnh train,
  5.000 ảnh val, 80 class, 221.790 vòng lặp tổng (7.393 vòng/epoch × 30).
- Mosaic tự tắt ở 10 epoch cuối (`--close-mosaic 10`), tức từ epoch 21.
- Thiết bị: `cuda`. Lúc đang chạy GPU dùng ~59%, ~8,1 GB VRAM: nút thắt vẫn là CPU đọc ảnh + mosaic, đúng như đã đo.
- File ra: `runs/coco-gpu/last.pt` (ghi mỗi epoch, ~51 MB, có optimizer), `runs/coco-gpu/best.pt` (trọng số EMA ở
  epoch mAP tốt nhất, ~12,8 MB), log `runs/coco-gpu.log`.

**Kết quả: xong 30/30 epoch, mAP50-95 tốt nhất 0.261 (mAP50 0.354 ở epoch 25; epoch 30: 0.353 / 0.261).**
Bắt đầu ~13:55 16/10, kết thúc 01:29 17/10: **~11,6 giờ** (dự kiến ban đầu 8 giờ).

**Số đo theo epoch** (từ `coco-gpu.log`)

| Epoch | box | cls | dfl | lr | Thời gian | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| 1 | 2.381 | 3.942 | 2.526 | 0.00033 | 954,7 s | | |
| 2 | 1.754 | 2.849 | 1.772 | 0.00067 | 877,6 s | | |
| 3 | 1.633 | 2.564 | 1.646 | 0.00100 | 917,2 s | | |
| 4 | 1.558 | 2.369 | 1.571 | 0.00100 | 918,3 s | | |
| 5 | 1.499 | 2.222 | 1.519 | 0.00099 | 964,0 s | 0.193 | 0.137 |
| 10 | 1.379 | 1.905 | 1.405 | 0.00084 | 932,3 s | 0.291 | 0.213 |
| 15 | 1.325 | 1.768 | 1.358 | 0.00059 | 1287,3 s | 0.319 | 0.235 |
| 20 | 1.287 | 1.669 | 1.327 | 0.00031 | 1794,9 s | 0.336 | 0.247 |
| 21 | 1.265 | 1.537 | 1.305 | 0.00026 | 1696,7 s | | (mosaic tắt) |
| 25 | 1.213 | 1.428 | 1.265 | 0.00009 | 1657,6 s | 0.354 | **0.261** |
| 30 | 1.192 | 1.384 | 1.251 | 0.00001 | 1653,0 s | 0.353 | **0.261** |

Epoch 6–9, 11–14, 16–19, 22–24, 26–29 không đo mAP (`--eval-every 5`); loss từng epoch xem trong log.

**Đánh giá**

- **Model dùng được ở mức yếu, chưa ngang bản gốc.** 0.261 bằng ~75% mốc 0.347 của `yolov8n.pt` gốc (cùng code đo,
  cùng val2017). Từ số 0 chỉ 30 epoch (bản gốc train 500) nên chênh lệch này là hợp lý; không phải bằng chứng lỗi
  code (model nạp trọng số gốc cho kết quả khớp Ultralytics, loss sai khác 0,0).
- **Chưa hội tụ.** mAP50-95 mỗi 5 epoch: 0.137 → 0.213 → 0.235 → 0.247 → 0.261 → 0.261. Phần lớn mức tăng cuối đến
  từ lúc tắt mosaic (epoch 21) và lr xuống thấp; epoch 25 → 30 không tăng thêm vì lr đã gần 0 (0.00009 → 0.00001).
  Loss vẫn còn giảm nhẹ (cls 1.428 → 1.384), nên train dài hơn sẽ còn lên. Không thể kết luận cần bao nhiêu epoch
  từ một lần chạy.
- **Chưa tách được tác dụng của từng thành phần** (EMA, mosaic, tắt mosaic ở 10 epoch cuối): chỉ có một lần chạy,
  không có đối chứng. Con số "tắt mosaic giúp" ở trên là quan sát về thời điểm, chưa phải kết luận nhân quả.
- `best.pt` ghi lúc 23:05 16/10 (epoch 25); epoch 30 bằng 0.261 ở 3 chữ số nên không thay `best.pt`. Chưa kiểm
  `best.pt` và `last.pt` khác nhau thế nào về mAP chính xác hơn 3 chữ số.
- Chưa so với job GCP (lần 2) vì chưa lấy được kết quả của nó.

**Hiện tượng bất thường: thời gian mỗi epoch tăng gần gấp đôi từ epoch 15–16.** Epoch 1–14: 878–986 s. Epoch 15:
1287 s. Epoch 16 trở đi: ~1640–1795 s, giữ nguyên tới epoch 30 (kể cả sau khi tắt mosaic ở epoch 21, nên không phải do
mosaic). Không có thay đổi cấu hình giữa chừng trong log. **Chưa biết nguyên nhân** (nghi ngờ máy nóng/giảm xung,
tiến trình khác chiếm CPU/đĩa, hoặc chế độ nguồn của Windows; chưa kiểm). Hệ quả: tổng 11,6 giờ thay vì ~8 giờ. Nếu
train lại, theo dõi nhiệt độ/xung GPU-CPU và tải CPU lúc tốc độ tụt.

**Sự cố / điều cần biết khi đọc log**

1. **Log chứa cả thanh tiến trình tqdm, nặng ~25 MB sau 7 epoch.** Lệnh chạy bằng PowerShell 5.1 với `>`: stderr
   (nơi tqdm ghi) bị gộp vào file, kèm một khối lỗi `NativeCommandError` ở đầu file. Ngược với ghi chú "log sạch khi
   chỉ chuyển stdout" trong nhật ký 16/10: ghi chú đó đúng với shell chuyển riêng stdout (bash, `cmd`, hoặc
   `2>$null`), không đúng với PowerShell 5.1.
2. **File log mã hoá UTF-16** (mặc định của `>` trong PowerShell 5.1), nên `grep`/`tail` thường không đọc được và
   tiếng Việt ở dòng đầu bị lỗi hiển thị. Đọc bằng PowerShell `Get-Content`, hoặc bằng Python:
   `open('coco-gpu.log', encoding='utf-16')`. Lọc dòng tóm tắt: dòng bắt đầu bằng `epoch  N/30  box`.
3. Lần sau nên chạy `... 2>$null | Out-File -Encoding utf8 runs/coco-gpu.log`, hoặc dùng `--log-every` thay cho
   chuyển hướng, để log gọn và đọc được bằng mọi công cụ. (Chưa đổi lệnh trong `quy-trinh-train.md`; xem Tiếp theo.)

**Tiếp theo**

- Ghi kết quả 0.261 / 0.354 vào `trang-thai.md` và `van-de-mo.md` (#8). Đã ghi vào đây; chưa sửa hai file đó.
- Đưa `best.pt` và `train.log` vào Release trên GitHub (file nặng không nằm trong git). Log nên chuyển UTF-8 và bỏ
  dòng tqdm trước khi đưa lên.
- Quyết: train thêm (resume từ `last.pt` đã có lr gần 0, nên cần đặt lại lịch lr) hay train lại dài hơn.
- Điều tra vì sao epoch chậm gấp đôi từ epoch 16 nếu sẽ train lại.
- Sửa `quy-trinh-train.md` mục 3.2 cho khớp ý 1–3 ở trên (log tqdm/UTF-16) và cập nhật thời gian thực tế.

---

## Lần 2: COCO từ số 0 trên GCP CPU

**Cấu hình:** VM `yolo-coco` (`n2-standard-8`, `us-east1-b`, ổ 150 GB), 30 epoch, `--lr 0.001`, `--eval-every 1`, log ở
`~/yolo/runs/coco-scratch/train.log`. Bắt đầu 05:06 UTC 15/10/2026. Dùng bản `load_mosaic` **cũ** (bản viết lại
ngày 15/10 chưa vào job này).

**Số đo:** ~0,146 s/ảnh, **~4,7 giờ/epoch**, đo mAP cả val2017 ~3,6 phút. Dự kiến xong ~21/10, ~56 USD credit. Loss
vòng 500 → 4000 của epoch 1: box 3.63 → 3.08, cls 5.70 → 4.91, dfl 4.26 → 3.41.

**Sự cố:** lệnh `wait` trong `prepare.sh` đợi luôn tiến trình `tee` của `exec > >(tee ...)` nên treo ~35 phút sau
khi tải COCO xong. Không dùng `exec > >(tee)` cùng `wait` không đối số.

**Chưa biết:** job còn chạy hay đã lỗi, đã tới epoch mấy, mAP bao nhiêu. Cần kiểm bằng
`gcloud compute ssh yolo-coco --zone=us-east1-b --command="tail -5 ~/yolo/runs/coco-scratch/train.log"` trên máy có
`gcloud`. **Bàn giao: ai tiếp nhận phải kiểm VM này, lấy kết quả, rồi xoá VM để khỏi tốn credit.**

---

## Lần 1: biển báo, fine-tune trên GCP CPU (lưu trữ)

**Cấu hình:** `andrewmvd/road-sign-detection` (702 ảnh train, 175 val, 4 class), fine-tune từ `yolov8n.pt`,
`n2-standard-8`, `us-central1-a`, 100 epoch, ~100 s/epoch, tổng ~2 giờ 50 phút, ~1,2 USD (ước tính). Chạy
11–12/10/2026, **chưa** có EMA và mosaic. VM và ổ đĩa đã xoá.

**mAP50-95 theo epoch:** 0 → 0.019; 10 → 0.598; 25 → 0.685; 50 → 0.702; **75 → 0.705 (best)**; 100 → 0.700. mAP50 tốt
nhất 0.836.

**Nhận xét:** từ epoch 25 mAP gần như đứng yên trong khi loss box vẫn giảm (0.48 → 0.08): bắt đầu học thuộc tập
train, 30–50 epoch là đủ. Vật nhỏ (~7×9 px) hay bị bỏ sót.

**Sự cố:**
- lr = 0.002 phá trọng số gốc sau vài bước (mAP về 0 dù loss vẫn giảm). Đổi sang công thức Ultralytics
  `0.002 × 5 / (4 + nc)` (QĐ-06).
- Tài khoản free trial không có hạn mức GPU nên phải train CPU (QĐ-08). `plink` lỗi với lệnh nhiều dòng nên gửi file
  script lên rồi chạy; `pip` của Debian quá cũ nên nâng `pip` trước.

Model lưu ở [Release v1.0](https://github.com/namnd020304/clone-python-yolo/releases/tag/v1.0) cùng `train.log`.

---

## Các thử nghiệm train nhỏ (không phải lần chạy chính)

| Thử nghiệm | Kết quả | Điều rút ra |
|---|---|---|
| Từ số 0 trên 4 ảnh coco8, 300 epoch | mAP50-95 0.995 (có và không EMA) | Vòng train, loss, assigner đúng |
| Cùng thử trên, mốc 50 / 100 / 150 / 200 epoch (có EMA) | 0.000 / 0.000 / 0.096 / 0.990 | mAP = 0 tới epoch 100 dù loss thấp; EMA không đổi hiện tượng này (van-de-mo #2) |
| Fine-tune coco8 30 epoch | 0.600 → 0.636 ở epoch 10, rồi giảm | Học thuộc 4 ảnh |
| Đo tốc độ COCO trên GPU, batch 16, 0 / 8 / 16 worker | 1,05 / 8,51 / 8,30 batch/s | Nút thắt là đọc ảnh + mosaic; chọn 8 worker |
| Cùng thử, batch 32, 8 worker | 4,27 batch/s (≈136 ảnh/s) | Batch lớn không nhanh hơn |
