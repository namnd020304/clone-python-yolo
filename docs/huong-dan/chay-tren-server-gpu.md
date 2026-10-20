# Chạy trên server GPU: train YOLOv8n tự viết và YOLOv8n gốc để so sánh

Viết cho người đem repo lên một máy Windows có GPU NVIDIA. Mục tiêu: train **từ số 0** trên COCO 2017 hai model với
cùng thông số rồi so mAP:

- `train.py`: YOLOv8n tự viết trong `yolo/`.
- `train_ultralytics.py`: YOLOv8n gốc của thư viện Ultralytics (`yolov8n.yaml`, trọng số ngẫu nhiên), đã đặt
  optimizer, lịch lr, augmentation và loss gain giống `train.py`.

Mọi lệnh là PowerShell, chạy từ thư mục gốc của repo (`clone-python-yolo`). Chi tiết thêm về `train.py` ở
[../ban-giao/quy-trinh-train.md](../ban-giao/quy-trinh-train.md).

## 0. Tóm tắt

```powershell
git clone https://github.com/namnd020304/clone-python-yolo.git; cd clone-python-yolo
uv sync; .venv\Scripts\activate
python check.py                                     # phải ra 16/16
# tải COCO vào ..\data\coco (mục 2), rồi:
mkdir runs -Force
python train.py --data ../data/coco --train-split train2017 --val-split val2017 --epochs 30 --batch 16 --workers 8 --lr 0.001 --eval-every 5 --out runs/coco-mine > runs/coco-mine.log
python train_ultralytics.py --data ../data/coco --train-split train2017 --val-split val2017 --epochs 30 --batch 16 --workers 8 --lr 0.001 --out runs/coco-ultra > runs/coco-ultra.log
```

## 1. Cài môi trường

Cần: driver NVIDIA (`nvidia-smi` chạy được), `git`, và `uv`. Chưa có `uv` thì cài rồi mở lại cửa sổ PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Cài thư viện đúng phiên bản trong `uv.lock` (Python 3.12, torch bản CUDA 12.8):

```powershell
uv sync
.venv\Scripts\activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python check.py            # phải in "Kết quả: 16/16"
```

Dòng `python -c` phải in `True` và tên GPU. In `False` thì train sẽ chạy trên CPU (chậm ~20 lần): kiểm tra
`nvidia-smi`, driver phải hỗ trợ CUDA ≥ 12.8 (driver ≥ 570). `check.py` không ra 16/16 thì **chưa train**.

`activate` báo lỗi "running scripts is disabled" thì chạy một lần
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` rồi thử lại.

## 2. Tải COCO 2017 (~26 GB, cần ~50 GB trống khi giải nén)

Dữ liệu đặt ngoài repo, ở `..\data\coco`:

```powershell
mkdir ..\data -Force; cd ..\data
curl.exe -L -o coco2017labels.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
curl.exe -L -o train2017.zip http://images.cocodataset.org/zips/train2017.zip      # 19 GB
curl.exe -L -o val2017.zip   http://images.cocodataset.org/zips/val2017.zip        # 0,8 GB
tar -xf coco2017labels.zip                       # tạo coco\labels\...
tar -xf train2017.zip -C coco/images
tar -xf val2017.zip   -C coco/images
(Get-ChildItem coco/images/train2017).Count      # 118287
(Get-ChildItem coco/labels/train2017).Count      # 117266 (ảnh không có vật nào thì không có file nhãn, đúng)
Remove-Item *.zip                                # lấy lại ~20 GB
cd ..\clone-python-yolo
```

**Không dùng `coco2017labels-segments.zip`**: nhãn đa giác sẽ bị đọc thành box sai mà không báo lỗi.

## 3. Chạy thử (1–2 phút, bắt buộc)

```powershell
python train.py --epochs 2 --eval-every 1 --batch 4 --out runs/thu
python train_ultralytics.py --epochs 1 --batch 4 --out runs/thu-ultra
```

Cả hai dùng `coco8` (4 ảnh, tự tải). Đạt khi chạy hết không lỗi và in ra mAP. mAP = 0 ở đây là bình thường.

## 4. Train thật

Train mất nhiều giờ, nên trước hết tắt chế độ ngủ (máy ngủ là dừng train):

```powershell
powercfg /change standby-timeout-ac 0      # train xong đặt lại, ví dụ 30 (phút)
```

Rồi chạy lần lượt (hoặc mỗi lệnh một cửa sổ PowerShell nếu có 2 GPU, xem dưới):

```powershell
mkdir runs -Force
python train.py --data ../data/coco --train-split train2017 --val-split val2017 `
    --epochs 30 --batch 16 --workers 8 --lr 0.001 --eval-every 5 --out runs/coco-mine > runs/coco-mine.log

python train_ultralytics.py --data ../data/coco --train-split train2017 --val-split val2017 `
    --epochs 30 --batch 16 --workers 8 --lr 0.001 --out runs/coco-ultra > runs/coco-ultra.log
```

- Hai lệnh **phải giữ cùng** `--epochs`, `--batch`, `--lr` thì mới so được.
- Có 2 GPU: mở 2 cửa sổ, cửa sổ đầu chạy `$env:CUDA_VISIBLE_DEVICES = "0"` trước lệnh `train.py`, cửa sổ sau chạy
  `$env:CUDA_VISIBLE_DEVICES = "1"` trước lệnh `train_ultralytics.py`. Chỉ 1 GPU thì chạy lần lượt.
- Đóng cửa sổ PowerShell là dừng train. Điều khiển server qua Remote Desktop thì ngắt kết nối (không Sign out) vẫn
  chạy tiếp.
- Thời gian tham khảo (RTX 5070 Ti, 28 luồng CPU): ~14,5 phút/epoch với `train.py`, tổng ~8 giờ cho 30 epoch. Nút
  thắt là CPU đọc ảnh, nên máy ít nhân CPU sẽ chậm hơn; khi đó đặt `--workers` bằng số nhân trừ 2.
- `train_ultralytics.py` mặc định chạy fp32 giống `train.py`. Thêm `--amp` thì nhanh hơn nhưng không còn cùng điều
  kiện.

Xem tiến độ từ cửa sổ PowerShell khác:

```powershell
Get-Content runs/coco-mine.log -Wait -Tail 5
Get-Content runs/coco-ultra/results.csv -Wait -Tail 3
nvidia-smi -l 5
```

### Bị ngắt giữa chừng

```powershell
# train.py: chạy lại đúng lệnh cũ, thêm --resume, đổi > thành >>
python train.py <tham số như cũ> --resume runs/coco-mine/last.pt >> runs/coco-mine.log
# train_ultralytics.py: chỉ cần
python train_ultralytics.py --resume runs/coco-ultra/weights/last.pt >> runs/coco-ultra.log
```

## 5. Kết quả ở đâu

| | YOLOv8n tự viết (`runs/coco-mine/`) | YOLOv8n Ultralytics (`runs/coco-ultra/`) |
|---|---|---|
| Trọng số | `best.pt`, `last.pt` | `weights/best.pt`, `weights/last.pt` |
| Chỉ số theo epoch | dòng `epoch ...` trong `runs/coco-mine.log`: loss train/val, mAP50, mAP50-95 (mỗi 5 epoch) | `results.csv`: loss train/val, precision, recall, mAP50, mAP50-95 (mỗi epoch) |
| Biểu đồ | không có | `results.png`, `BoxPR_curve.png`, `confusion_matrix.png` |
| Bảng theo class | không có | cuối `runs/coco-ultra.log` |

`runs/` và file `.pt` không nằm trong git (`.gitignore`), nên muốn đem về máy khác thì nén thư mục lại rồi copy:

```powershell
Compress-Archive runs/coco-mine, runs/coco-ultra, runs/coco-mine.log, runs/coco-ultra.log ket-qua.zip
```

## 6. So sánh cho công bằng

Mỗi bên tự tính mAP bằng code riêng, nên số có thể lệch vài phần nghìn chỉ vì cách tính. Để chắc, hãy chấm cả hai
`best.pt` trên val2017 bằng **cùng một** bộ đo: nạp `best.pt` của Ultralytics vào `YOLOv8n` qua
`load_from_ultralytics` (giống `compare.py`) rồi chạy `yolo.metrics.evaluate`. So trực quan trên vài ảnh:

```powershell
python compare.py runs/coco-mine/best.pt runs/coco-ultra/weights/best.pt datasets/coco8/images/val/000000000036.jpg
```

Những khác biệt còn lại giữa hai lần train (không chỉnh được bằng tham số): Ultralytics đổi lr theo epoch, `train.py`
đổi theo từng vòng lặp; khi đo, Ultralytics letterbox kiểu `rect` (đệm viền ít hơn). Cả hai ảnh hưởng nhỏ.
