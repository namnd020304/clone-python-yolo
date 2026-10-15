# Vận hành: các lệnh thường dùng

Mọi lệnh chạy từ thư mục gốc repo, với `venv` đã activate (xem `cai-dat-moi-truong.md`).

## Bảng lệnh

| Lệnh | Làm gì | Bước |
|---|---|---|
| `python check.py [N ...]` | Chấm 16 bài (hoặc chỉ bài N) | 1–16 |
| `python load_weights.py [anh.jpg]` | Nạp `yolov8n.pt` gốc, so đầu ra với Ultralytics, vẽ `result.jpg` | 9+ |
| `python kaggle_data.py <owner/dataset> [--name ten]` | Tải dataset Kaggle, đổi về cấu trúc chuẩn trong `datasets/` | 2b |
| `python train.py [...]` | Train / fine-tune, lưu `runs/<out>/last.pt`, `best.pt` | 14 |
| `python predict.py <ckpt> <anh>` | Dự đoán một ảnh, vẽ `predict.jpg` | 2b |
| `python webcam.py [...]` | Nhận diện từ camera, nhấn `q` hoặc `Esc` để thoát | 2b |

Khi thêm hoặc đổi lệnh, cập nhật bảng này và mục chi tiết bên dưới.

## Lấy dữ liệu: `kaggle_data.py`

```powershell
python kaggle_data.py andrewmvd/road-sign-detection --name bien-bao   # mã = phần cuối URL kaggle.com/datasets/...
python kaggle_data.py D:/thu_muc_da_tai --name ten                    # thư mục đã có trên máy
```

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `source` | Mã dataset Kaggle, hoặc thư mục trên máy | bắt buộc |
| `--name` | Tên thư mục trong `datasets/` | phần sau dấu `/` |
| `--out` | Thư mục cha | `datasets` |
| `--val` | Tỉ lệ val khi dataset chưa chia sẵn | 0.2 |
| `--seed` | Hạt giống chia ngẫu nhiên | 0 |

Kết quả:

```
datasets/<tên>/
├── images/train/*.jpg, images/val/*.jpg
├── labels/train/*.txt, labels/val/*.txt     mỗi dòng: class cx cy w h (chia cho rộng/cao ảnh, 0..1)
└── classes.txt                              dòng i = tên class i
```

- Nhận ra YOLO `.txt` (kể cả kiểu Roboflow `train/images` + `valid/images` + `data.yaml`) và Pascal VOC `.xml`.
- Chưa hỗ trợ: COCO JSON, CSV, nhãn đa giác (segmentation).
- Dataset công khai tải không cần tài khoản. Dataset cần đăng nhập: đặt `kaggle.json` ở `C:/Users/<bạn>/.kaggle/`.
- Tự gán nhãn ảnh của mình: Roboflow, CVAT, Label Studio, export "YOLO" / "YOLOv8".

## Train: `train.py`

```powershell
python train.py --weights yolov8n.pt --epochs 30                              # fine-tune trên coco8
python train.py --data datasets/bien-bao --weights yolov8n.pt --epochs 50     # dataset riêng
python train.py --epochs 300 --lr 0.002 --val-split train --no-augment        # phép thử học thuộc từ số 0
```

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `--data` | Thư mục có `images/<split>`, `labels/<split>`; `coco8` thì tự tải | `coco8` |
| `--train-split`, `--val-split` | Tên thư mục con | `train`, `val` |
| `--nc` | Số class; 0 = đọc `classes.txt`, không có thì 80 | 0 |
| `--weights` | `yolov8n.pt` hoặc checkpoint của `train.py`; bỏ trống = train từ đầu | trống |
| `--resume` | `last.pt` của lần chạy bị ngắt; các tham số khác giữ như cũ | trống |
| `--epochs`, `--batch`, `--size` | | 100, 8, 640 |
| `--lr` | 0 = tự tính `0.002 × 5 / (4 + nc)` (QĐ-06) | 0 |
| `--weight-decay` | | 5e-4 |
| `--no-augment` | Tắt mọi augmentation | tắt |
| `--mosaic`, `--close-mosaic` | Xác suất mosaic; tắt mosaic ở N epoch cuối | 1.0, 10 |
| `--no-ema` | Không dùng EMA | tắt |
| `--eval-every` | Đo mAP mỗi N epoch (và epoch cuối) | 10 |
| `--workers` | Số tiến trình đọc dữ liệu | 0 |
| `--out` | Nơi lưu `last.pt`, `best.pt` | `runs/train` |

- `last.pt` ghi đè **mỗi epoch** (kèm optimizer, EMA, vị trí lr để resume); `best.pt` ở epoch đo mAP khi mAP50-95
  tốt hơn. Muốn ghi log ra file:
  `python train.py ... > runs/ten/train.log`.
- Thời gian (CPU máy nhà, 702 ảnh, size 640): ~2,5–3 phút/epoch kể cả đo mAP. `--size 416` nhanh hơn khoảng 2 lần.
- Dataset ~700 ảnh: mAP gần như đứng yên sau epoch 25–50.
- Bị ngắt giữa chừng: chạy lại đúng lệnh cũ, thêm `--resume runs/<ten>/last.pt`. Chạy tiếp từ epoch sau epoch
  cuối đã lưu, giữ optimizer, EMA và lịch lr. Thứ tự xáo dữ liệu không giống hệt lần chạy liền một mạch.

## Dự đoán: `predict.py`

```powershell
python predict.py runs/bien-bao/best.pt datasets/bien-bao/images/val/road160.png
python predict.py runs/bien-bao/best.pt anh.jpg --conf 0.1 --out ket_qua.jpg
```

In tên class, điểm, toạ độ; vẽ vào `--out` (mặc định `predict.jpg`). `--conf` mặc định 0.25; hạ xuống 0.1 để bắt vật
nhỏ điểm thấp. Không cần Ultralytics. Model biển báo chỉ biết 4 class (crosswalk, speedlimit, stop, trafficlight).

## Camera: `webcam.py`

```powershell
python webcam.py                                          # chỉ hiện người, model COCO yolov8n_mine.pt
python webcam.py --size 320                               # nhanh hơn (~9,5 FPS trên CPU máy nhà)
python webcam.py --all                                    # mọi vật COCO
python webcam.py --weights runs/bien-bao/best.pt --all    # model biển báo
```

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `--weights` | Checkpoint định dạng của repo | `yolov8n_mine.pt` |
| `--camera` | Chỉ số camera | 0 |
| `--conf` | Ngưỡng điểm | 0.4 |
| `--size` | Cạnh ảnh đưa vào model, chia hết cho 32 | 640 |
| `--all` | Hiện mọi class (tự bật khi model không phải 80 class) | tắt |
| `--no-mirror` | Không lật ngang như gương | tắt |

Model nhận ra "có một người", không biết người đó là ai (nhận diện khuôn mặt là bài toán khác).

## Train trên máy ảo GCP (CPU, qua đêm)

Đã làm một lần ngày 11–12/10 (QĐ-08). Các bước:

1. Chọn đúng tài khoản và project sẽ trả tiền: `gcloud config list`. Kiểm tra trước, máy có thể đăng nhập nhiều tài khoản.
2. Tạo VM `n2-standard-8` ở `us-central1-a`, image Debian.
3. Đóng gói code **không gồm** `venv/`, `datasets/`, `runs/`; `gcloud compute scp` lên VM.
4. Trên VM: nâng `pip` trước (pip của Debian cũ), cài thư viện, chạy `kaggle_data.py` ngay trên VM.
5. Chạy `train.py ... > runs/<ten>/train.log` trong `tmux`, nối thêm lệnh tự tắt VM khi xong.
   Trên Windows, `plink` lỗi với lệnh nhiều dòng: viết các bước vào một file `.sh`, gửi lên rồi chạy file đó.
6. Xem tiến độ: `gcloud compute ssh yolo-train --zone=us-central1-a --command="tail -5 ~/yolo/runs/<ten>/train.log"`.
7. Lấy kết quả rồi **xoá VM** (ổ đĩa vẫn tính tiền khi VM đã tắt):

```powershell
$G = "--zone=us-central1-a"
gcloud compute instances start yolo-train $G
gcloud compute ssh yolo-train $G --command="tar -czf runs.tgz -C ~/yolo runs"
gcloud compute scp yolo-train:runs.tgz . $G
tar -xzf runs.tgz
gcloud compute instances delete yolo-train $G --quiet
```

Chi phí lần trước: ~0,39 USD/giờ × ~2 giờ 50 phút ≈ 1,2 USD credit (ước tính).

## Nơi lưu kết quả chạy

| Thứ | Ở đâu |
|---|---|
| Dataset | `datasets/<tên>/` |
| Checkpoint, log train | `runs/<out>/last.pt`, `best.pt`, `train.log` |
| Ảnh dự đoán | `predict.jpg`, `result.jpg`, `runs/<tên>/pred_*.jpg` |
| Ảnh kiểm tra mosaic | `runs/mosaic_check.jpg` |

Tất cả nằm ngoài git. Số đo quan trọng chép vào `ban-giao/nhat-ky.md` và bảng kết quả trong `ban-giao/trang-thai.md`.
