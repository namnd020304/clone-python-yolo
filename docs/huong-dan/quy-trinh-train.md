# Quy trình train đầy đủ: từ bài toán đến model phát hành

File này là **checklist** cho một lần train nghiêm túc (dataset riêng, chạy lâu, kết quả để dùng hoặc báo cáo). Đi
đúng thứ tự giai đoạn 0 → 11; mỗi giai đoạn có **đầu ra** và **điều kiện qua cửa**: chưa qua cửa thì chưa sang giai
đoạn sau. Lý thuyết từng mảnh ở [../train/](../train/README.md), lệnh và tham số ở [van-hanh.md](van-hanh.md).

Ký hiệu cột "Repo": ✅ đã có công cụ · ⚠️ có một phần · ❌ chưa có, phải làm tay hoặc viết thêm (xem mục 13).

```
0 Bài toán ─► 1 Môi trường ─► 2 Dữ liệu ─► 3 Chia tập ─► 4 Sanity check ─► 5 Cấu hình
                                                                              │
11 Ghi chép ◄─ 10 Phát hành ◄─ 9 Ngưỡng & tốc độ ◄─ 8 Đánh giá ◄─ 7 Theo dõi ◄─ 6 Chạy train
      │
      └──► 12 Vòng cải thiện: lỗi phân tích ở bước 8 → sửa dữ liệu / cấu hình → quay lại bước 2 hoặc 5
```

## 0. Xác định bài toán và tiêu chí thành công

Làm **trước khi** đụng vào dữ liệu. Không có tiêu chí thì không biết khi nào dừng.

| Việc | Ví dụ (biển báo) | Repo |
|---|---|---|
| Danh sách class, định nghĩa rõ từng class (cái gì tính, cái gì không) | 4 class: crosswalk, speedlimit, stop, trafficlight | ❌ ghi tay |
| Ảnh lúc dùng thật trông thế nào: camera, độ phân giải, góc, ánh sáng, ngày/đêm | Camera hành trình, biển ở xa | ❌ |
| Vật nhỏ nhất cần bắt được (pixel trên ảnh gốc) | ~10 px (hiện đang bỏ sót, van-de-mo #3) | ❌ |
| Chỉ số chính và mức chấp nhận | mAP50-95 val ≥ 0.705 (mốc hiện có); recall class `stop` ≥ … | ❌ |
| Ràng buộc chạy: CPU/GPU, FPS tối thiểu, `--size` | CPU máy nhà: ~5 FPS ở 640, ~9,5 FPS ở 320 | ✅ đo được bằng `webcam.py` |
| Ngân sách: giờ máy, USD credit | ~1,2 USD / 100 epoch trên `n2-standard-8` | ✅ |

**Đầu ra:** một đoạn trong `nhat-ky.md` ghi các mục trên. **Qua cửa:** có con số mục tiêu cụ thể.

## 1. Môi trường và khả năng tái lập

| Việc | Lệnh / cách làm | Repo |
|---|---|---|
| Cài môi trường | [cai-dat-moi-truong.md](cai-dat-moi-truong.md) | ✅ |
| Chấm toàn bộ code | `python check.py` → 16/16 | ✅ |
| Code sạch, đã commit | `git status` trống; ghi lại commit hash sẽ dùng để train | ⚠️ hash chưa tự lưu vào checkpoint |
| Ghi phiên bản `torch`, Python, máy (CPU/GPU) | `python -c "import torch; print(torch.__version__)"` → chép vào log | ❌ |
| Seed cố định | `train.py` đặt `torch.manual_seed(0)` | ✅ |
| Lưu nguyên câu lệnh train | Dòng đầu `train.log`; hoặc file `runs/<ten>/cmd.txt` | ❌ ghi tay |

**Qua cửa:** `check.py` 16/16 **trên đúng máy sẽ train** (VM mới cũng phải chấm lại, xem 07-du-lieu-that-va-cloud.md).

## 2. Dữ liệu: lấy, chuyển định dạng, kiểm tra chất lượng

Lỗi dữ liệu phổ biến hơn lỗi code nhiều. Giai đoạn này đáng thời gian nhất.

### 2.1 Lấy và chuyển định dạng

| Việc | Lệnh / cách làm | Repo |
|---|---|---|
| Tải từ Kaggle, đổi YOLO/VOC → cấu trúc chuẩn | `python kaggle_data.py <owner/ds> --name <ten>` | ✅ |
| Dataset COCO JSON / CSV / đa giác | Viết thêm bộ đọc | ❌ |
| Tự gán nhãn ảnh của mình | Roboflow / CVAT / Label Studio, export "YOLO" | ✅ (công cụ ngoài) |
| Quy tắc gán nhãn viết thành văn bản | Box ôm sát? Vật bị che >50% có gán không? Vật quá nhỏ? | ❌ |

### 2.2 Kiểm tra từng file

| Lỗi cần bắt | Cách kiểm | Repo |
|---|---|---|
| Ảnh hỏng / không mở được | Mở thử mọi ảnh bằng PIL | ❌ |
| Ảnh có nhãn EXIF xoay | So `Image.open(p).size` với ảnh sau `ImageOps.exif_transpose` | ❌ |
| Dòng nhãn sai số cột, class id ≥ nc hoặc < 0 | Đọc mọi file `.txt` | ❌ |
| Toạ độ ngoài [0, 1], `w` hoặc `h` ≤ 0 | Như trên | ❌ |
| Box trùng lặp (cùng class, IoU > 0.9 trong một ảnh) | `box_iou` từ `yolo/postprocess.py` | ❌ |
| Ảnh không có file nhãn (ảnh nền) — cố ý hay sót? | Đếm; ảnh nền cố ý nên giữ ~0–10% | ⚠️ dataset đọc được, chưa đếm |
| Nhãn đúng vật | **Vẽ nhãn lên ≥ 20 ảnh ngẫu nhiên mỗi split và nhìn** | ⚠️ đã làm tay một lần |

### 2.3 Thống kê

| Thống kê | Vì sao cần | Repo |
|---|---|---|
| Số ảnh, số box của từng class ở từng split | Class quá ít box sẽ kém; mất cân bằng | ❌ |
| Phân bố kích thước box (pixel sau letterbox 640): nhỏ < 32², vừa 32²–96², lớn > 96² | Biết bao nhiêu vật nhỏ; chọn `--size` | ❌ |
| Số box mỗi ảnh (trung bình, lớn nhất) | Ảnh rất đông vật cần chú ý khi mosaic / NMS | ❌ |
| Tỉ lệ khung ảnh, độ phân giải gốc | Ảnh gốc nhỏ thì phóng lên 640 không thêm thông tin | ❌ |

**Đầu ra:** bảng thống kê trong `nhat-ky.md`. **Qua cửa:** 0 lỗi định dạng; đã nhìn ảnh có nhãn; biết class nào
ít dữ liệu, tỉ lệ vật nhỏ bao nhiêu.

## 3. Chia train / val / test

| Việc | Ghi chú | Repo |
|---|---|---|
| Ba tập riêng: **train** (học), **val** (chọn epoch, chỉnh cấu hình), **test** (đo **một lần** ở cuối) | Chọn cấu hình theo val nhiều lần là "học" cả val; số trên val lạc quan dần | ❌ `kaggle_data.py` chỉ ra train/val, và **bỏ** split `test` khi dataset đã có `val` |
| Tỉ lệ gợi ý | 70 / 15 / 15 khi < 2.000 ảnh; giữ split gốc nếu dataset đã chia | ⚠️ `--val 0.2` |
| Chia theo nhóm, không theo ảnh | Khung hình cùng video, ảnh chụp liên tiếp cùng chỗ phải nằm cùng một tập | ❌ |
| Kiểm tra trùng giữa các tập | Hash ảnh (giống hệt) và hash cảm nhận (gần giống) giữa train ↔ val ↔ test | ❌ |
| Mỗi class đều có mặt ở cả 3 tập | Phân tầng (stratify) theo class hiếm nhất | ❌ |
| Chốt và không đổi | Seed cố định; ghi số ảnh từng tập. Đổi cách chia = mọi số đo cũ không còn so được | ✅ `--seed 0` |

**Qua cửa:** 3 tập không trùng, mỗi class có đủ box ở mỗi tập (gợi ý ≥ 20 box ở val và test).

## 4. Sanity check trước khi train dài

Mỗi bước chỉ vài phút, chặn trước được cả đêm train vô ích.

| # | Kiểm tra | Lệnh | Kỳ vọng | Repo |
|---|---|---|---|---|
| 4.1 | Vẽ một batch train **sau augmentation** (mosaic, lật, sáng) kèm box | như `runs/mosaic_check.jpg` | Box khớp vật, không lệch sau lật | ⚠️ đã làm tay |
| 4.2 | mAP của trọng số trước train | `train.py --weights ...` in "Trước khi train" | Fine-tune class mới: ~0 (nhánh class khởi tạo lại) | ✅ |
| 4.3 | Phép thử học thuộc trên coco8 | `python train.py --epochs 300 --lr 0.002 --val-split train --no-augment` | mAP50-95 ≈ 0.995 | ✅ |
| 4.4 | Học thuộc **16–32 ảnh của chính dataset** (val = chính các ảnh đó, không aug) | tạo thư mục con, `--val-split train --no-augment` | mAP50-95 > 0.9; nếu không: nghi nhãn hoặc class id | ❌ chưa làm với dataset thật |
| 4.5 | Chạy 1 epoch đủ dataset, `--log-every 20` | | Loss giảm trong epoch, không NaN; đo thời gian/epoch | ✅ |
| 4.6 | Thử `--resume` | Ngắt sau 1–2 epoch, chạy lại với `--resume` | Chạy tiếp đúng epoch, lr liền mạch | ✅ |
| 4.7 | Ước tính tổng thời gian và chi phí | thời gian/epoch × epoch × giá máy | Trong ngân sách bước 0 | ✅ tính tay |

## 5. Chọn cấu hình

Đổi **một** thứ mỗi lần so với mốc đang có. Mọi giá trị dưới đây là mặc định hiện tại của `train.py`.

| Thứ | Mặc định | Khi nào đổi |
|---|---|---|
| Khởi đầu | `--weights yolov8n.pt` (fine-tune) | Từ số 0 chỉ khi dataset rất lớn (COCO) hoặc miền ảnh rất khác |
| `--size` | 640 | Vật nhỏ nhiều → thử 800/960 (chậm hơn ~1,6/2,3 lần); cần nhanh → 416/320 |
| `--epochs` | 100 | ~700 ảnh: mAP đứng yên sau 25–50 epoch (đo 12/10) |
| `--batch` | 8 | GPU thì tăng; nhớ lr AdamW không tự scale theo batch |
| `--lr` | `0.002 × 5 / (4 + nc)` (QĐ-06) | Từ số 0 job dài: thử lớn hơn (COCO đang chạy `0.001`) |
| `--weight-decay` | 5e-4 | Overfit rõ → tăng nhẹ |
| `--mosaic`, `--close-mosaic` | 1.0, 10 | Chưa chứng minh có lợi trên `bien-bao` (van-de-mo #1) |
| EMA | bật | Như trên |
| HSV, xoay, co giãn | **chưa có** | Plan mục 6.2 |
| `--eval-every` | 10 | Dataset nhỏ: 1–5 để không lỡ epoch tốt nhất |
| `--workers` | 0 | Máy nhiều nhân: 4–8 |

**Đầu ra:** câu lệnh đầy đủ ghi vào `nhat-ky.md` **trước khi chạy**, kèm giả thuyết "kỳ vọng tốt hơn mốc X vì Y".

## 6. Chạy train

| Việc | Cách làm | Repo |
|---|---|---|
| Chạy máy nhà | `python -u train.py ... 2>&1 \| tee runs/<ten>/train.log` | ✅ |
| Chạy máy ảo | Theo [van-hanh.md](van-hanh.md) mục GCP: `tmux`, log ra file, tự tắt VM khi xong | ✅ |
| Kiểm tra tài khoản / project / vùng trước khi tạo VM | `gcloud config list` | ✅ |
| Không ghi đè lần chạy cũ | `--out runs/<ten>-<ngay>-<thay-doi>` | ⚠️ đặt tên tay |
| Bị ngắt | Đúng lệnh cũ + `--resume runs/<ten>/last.pt` | ✅ |
| Sao lưu giữa chừng (job nhiều ngày) | Kéo `last.pt` về máy định kỳ | ⚠️ làm tay |
| Xong: lấy `runs/` về rồi **xoá VM** | Ổ đĩa vẫn tính tiền khi VM tắt | ✅ |

## 7. Theo dõi và chẩn đoán trong lúc train

Xem log sau epoch 1, epoch 10 và mỗi lần đo mAP.

| Triệu chứng | Nguyên nhân hay gặp | Làm gì |
|---|---|---|
| Loss NaN / nổ | lr quá lớn, box lỗi (w = 0) | Giảm lr; chạy lại kiểm tra 2.2 |
| Loss không giảm ngay từ đầu | lr quá nhỏ; nhãn không đọc được (mọi ảnh 0 box) | Kiểm tra 4.4 |
| Fine-tune: mAP tụt về 0 sau vài bước | lr quá lớn phá trọng số gốc (đã gặp 11/10) | Dùng lr mặc định QĐ-06 |
| Loss giảm, mAP = 0 lâu (từ số 0) | Hiện tượng đã thấy, chưa giải thích (van-de-mo #2) | Train tiếp; đo IoU, điểm class theo epoch |
| Loss train giảm, mAP val đứng yên / giảm | Overfit | Dừng sớm; thêm dữ liệu / augmentation |
| mAP50 cao, mAP50-95 thấp | Box chưa khít | Train lâu hơn, `--size` lớn hơn |
| Thời gian/epoch tăng dần | Máy khác tranh CPU, ổ đầy | `top`, `df -h` |

Dừng sớm: chưa có `--patience` tự động. Quy tắc tay: mAP50-95 val không tăng quá 0.005 qua 3 lần đo liên tiếp thì
dừng (`best.pt` đã được giữ).

## 8. Đánh giá

Đánh giá trên **val** để chọn; trên **test** để báo cáo, **một lần**, bằng `best.pt` đã chốt.

| Chỉ số / phân tích | Vì sao | Repo |
|---|---|---|
| mAP50, mAP50-95 tổng | Chỉ số chính, so được với Ultralytics | ✅ `yolo/metrics.py` |
| Đánh giá một checkpoint trên một split bất kỳ, không cần train | Đo test, đo lại checkpoint cũ | ❌ chưa có `val.py` |
| AP **từng class**, số box từng class | Biết class nào kéo điểm xuống | ❌ `mean_ap` chỉ trả trung bình |
| Precision, recall ở ngưỡng dùng thật (ví dụ conf 0.25) | mAP không cho biết model bỏ sót bao nhiêu khi chạy thật | ❌ |
| mAP theo kích thước vật: nhỏ / vừa / lớn | Điểm yếu vật nhỏ (van-de-mo #3) cần con số, không chỉ cảm nhận | ❌ |
| Ma trận nhầm lẫn (gồm cột "nền") | Nhầm class với nhau hay bỏ sót | ❌ |
| Xem lỗi bằng mắt: ảnh FP điểm cao nhất, ảnh FN | Phát hiện nhãn sai trong val, kiểu lỗi lặp lại | ⚠️ `predict.py` từng ảnh |
| Độ dao động | Val 175 ảnh: chênh 0.01 có thể là nhiễu. So 2 cấu hình nên chạy ≥ 2 seed | ❌ chưa có `--seed` cho train |

**Qua cửa:** đạt tiêu chí bước 0 trên val; có bảng từng class; đã xem ≥ 20 ảnh lỗi.

## 9. Chọn ngưỡng và tốc độ suy luận

| Việc | Ghi chú | Repo |
|---|---|---|
| Chọn `--conf` theo nhu cầu (ưu tiên không bỏ sót hay không báo nhầm) | Quét conf 0.1 → 0.6 trên val, xem P/R | ❌ |
| NMS IoU | Mặc định 0.7 lúc đo mAP; 0.45–0.7 lúc dùng | ✅ tham số có sẵn |
| Đo tốc độ trên **máy sẽ chạy thật**, từng `--size` | ms/khung, FPS | ✅ `webcam.py` |
| Đổi `--size` lúc chạy → đo lại mAP ở size đó | Size 320 bỏ sót người trên `bus.jpg` (12/10) | ⚠️ |

## 10. Đóng gói và phát hành

| Việc | Repo |
|---|---|
| `best.pt` chứa: trọng số EMA, `nc`, `names`, `epoch`, `map` | ✅ |
| Kèm: câu lệnh/tham số, commit hash, phiên bản torch, thống kê dataset | ❌ phải lưu thêm vào checkpoint hoặc file bên cạnh |
| `train.log` + bảng kết quả (val, test, từng class) | ⚠️ |
| Thẻ model (model card): dữ liệu, chỉ số, giới hạn đã biết (vật < 10 px…) | ❌ |
| Đưa lên GitHub Release (không commit `.pt` vào git) | ✅ đã làm v1.0 |
| Xuất TorchScript / ONNX cho C++ | ❌ giai đoạn 3 |
| Kiểm tra lại sau khi tải về: `predict.py` trên vài ảnh val ra đúng như trước | ✅ |

## 11. Ghi chép (theo [quy-trinh-lam-viec.md](quy-trinh-lam-viec.md))

- `ban-giao/nhat-ky.md`: lệnh, giả thuyết, số đo theo epoch, chi phí, lỗi gặp.
- `ban-giao/trang-thai.md`: dòng mới trong bảng kết quả đo (có ngày).
- `kien-truc/quyet-dinh.md`: nếu đổi mặc định (ví dụ tắt mosaic cho dataset nhỏ).
- `ban-giao/van-de-mo.md`: đóng hoặc mở vấn đề.
- Commit tài liệu; **không** commit `runs/`, `datasets/`, `*.pt`.

## 12. Vòng cải thiện

Sau bước 8, chọn **một** hướng theo loại lỗi chiếm nhiều nhất, rồi quay lại:

| Lỗi chính | Hướng | Quay lại bước |
|---|---|---|
| Nhãn sai / thiếu trong dữ liệu | Sửa nhãn | 2 |
| Một class kém, ít box | Thêm ảnh class đó | 2 |
| Bỏ sót vật nhỏ | `--size` lớn hơn, co giãn ngẫu nhiên, ảnh có vật nhỏ | 5 hoặc 2 |
| Báo nhầm trên nền | Thêm ảnh nền (không vật) | 2 |
| Overfit sớm | Augmentation mạnh hơn (HSV, xoay), dừng sớm | 5 |
| Box chưa khít | Train lâu hơn, size lớn hơn | 5 |

Mỗi vòng là một dòng trong bảng so sánh: `tên run · thay đổi duy nhất · mAP50 · mAP50-95 · ghi chú`.

## 13. Khoảng trống của repo để có quy trình trên đủ công cụ

Xếp theo thứ tự nên làm. Mỗi mục là một bước mới: có bài chấm trong `check.py` nếu là code (QĐ-02).

| # | Việc | File | Phục vụ bước |
|---|---|---|---|
| 1 | Script kiểm tra + thống kê dataset (lỗi 2.2, bảng 2.3, vẽ nhãn ngẫu nhiên) | `data_check.py` mới | 2 |
| 2 | Giữ split `test`; thêm `--test` để tự tách test; kiểm tra ảnh trùng giữa các tập | `kaggle_data.py` | 3 |
| 3 | `val.py`: đánh giá checkpoint trên split bất kỳ | mới | 8 |
| 4 | Metrics mở rộng: AP từng class, P/R ở một ngưỡng, mAP theo kích thước, ma trận nhầm lẫn | `yolo/metrics.py` | 8, 9 |
| 5 | `train.py`: `--seed`, `--patience`, lưu tham số + commit hash + phiên bản vào checkpoint, ghi `results.csv` mỗi epoch | `train.py` | 1, 7, 10 |
| 6 | Augmentation HSV, xoay / co giãn (plan 6.2) | `yolo/dataset.py` | 5, 12 |
| 7 | Chạy đo EMA + mosaic trên `bien-bao` theo đúng quy trình này (van-de-mo #1) | — | 5–8 |
| 8 | AMP khi có GPU; xuất TorchScript | `train.py`, mới | 6, 10 |
