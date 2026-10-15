# Cài đặt môi trường

Mục tiêu: từ máy trắng đến khi `python check.py` báo **Kết quả: 16/16**. Viết cho Windows 11 (máy dev hiện tại);
macOS/Linux làm tương tự, đổi `venv\Scripts\` thành `venv/bin/`.

## 1. Công cụ cần cài

| Công cụ | Phiên bản đã dùng | Ghi chú |
|---|---|---|
| Python | 3.10.11 | `C:\Users\<bạn>\AppData\Local\Programs\Python\Python310`; 3.10 trở lên đều được |
| git | — | |
| Google Cloud SDK (`gcloud`) | — | Chỉ cần khi train trên máy ảo GCP |
| Camera | — | Chỉ cần cho `webcam.py` |

Không cần GPU. Mọi thứ trong repo chạy được trên CPU (train chậm, xem số đo ở `ban-giao/trang-thai.md`).

## 2. Lấy code và cài thư viện

```powershell
git clone https://github.com/namnd020304/clone-python-yolo.git
cd clone-python-yolo
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchvision numpy pillow     # bắt buộc: model, train, check.py
python -m pip install ultralytics                       # đối chiếu: load_weights.py, check.py bài 12-13, tải yolov8n.pt
python -m pip install opencv-python                     # webcam.py
python -m pip install kagglehub                         # kaggle_data.py
```

Phiên bản đã chạy được (12/10/2026): `torch 2.14.1` (CPU), `torchvision 0.29.1`, `numpy 2.2.6`, `pillow 12.3.0`,
`ultralytics 8.4.175`, `opencv-python 5.0.0.93`, `kagglehub 1.0.2`. Repo chưa có `requirements.txt`; nếu thêm thì ghi
đúng các bản này.

Các lệnh trong tài liệu viết là `python ...` với `venv` đã activate. Không activate thì gọi thẳng
`venv\Scripts\python.exe ...`.

## 3. Trọng số và dữ liệu

| File | Lấy thế nào | Dùng cho |
|---|---|---|
| `yolov8n.pt` | Tự tải khi chạy `load_weights.py` (Ultralytics), hoặc tải tay: `curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt` | Đối chiếu, điểm xuất phát fine-tune |
| `yolov8n_mine.pt` | Đổi từ `yolov8n.pt` sang định dạng checkpoint của repo (lệnh bên dưới; in "Nạp 354/355 tensor") | `webcam.py` mặc định |
| `datasets/coco8` | Tự tải khi chạy `train.py` với `--data coco8` (mặc định) | Phép thử nhanh |
| `datasets/bien-bao` | `python kaggle_data.py andrewmvd/road-sign-detection --name bien-bao` | Dataset thật 4 class |

Tạo `yolov8n_mine.pt` (cần `yolov8n.pt` ở gốc repo và đã cài ultralytics):

```powershell
python -c "import torch; from train import load_pretrained; from yolo.model import YOLOv8n; from yolo.dataset import COCO_NAMES; m = YOLOv8n(); load_pretrained(m, 'yolov8n.pt'); torch.save({'model': m.state_dict(), 'nc': 80, 'names': COCO_NAMES}, 'yolov8n_mine.pt')"
```

Tất cả nằm ngoài git (`.gitignore` chặn `*.pt`, `datasets/`, `runs/`).

## 4. Kiểm tra

```powershell
python check.py            # 16 bài; bài 12-13 cần ultralytics
python check.py 7          # chỉ chấm một bài
python load_weights.py     # so với Ultralytics, ra result.jpg: 4 người + 1 xe buýt, lệch ~6e-05
```

## 5. Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `OSError: [WinError 4551] An Application Control policy has blocked this file ... torch\lib\*.dll` | Windows (Smart App Control / chính sách ứng dụng) chặn DLL của torch. Đã gặp trong terminal của trợ lý AI (11/10, 15/10); terminal của người dùng thì chạy được | Chạy lệnh trong terminal VS Code / PowerShell của mình. Nếu vẫn bị chặn: kiểm tra Smart App Control trong Windows Security |
| `UnicodeEncodeError: 'charmap' ...` khi in tiếng Việt | Console Windows dùng bảng mã cũ | `check.py` đã tự chuyển UTF-8. Script khác: đặt `PYTHONUTF8=1` |
| `ModuleNotFoundError: ultralytics` ở bài 12–13 | Chưa cài ultralytics | `python -m pip install ultralytics` |
| `ModuleNotFoundError: cv2` | Chưa cài OpenCV | `python -m pip install opencv-python` |
| mAP về 0 ngay sau vài bước fine-tune | lr quá lớn | Để `--lr 0` (mặc định, tự tính) (QĐ-06) |
| `FileNotFoundError: không có ảnh trong ...` | `--data` sai, hoặc thư mục val tên `valid` | Xem cấu trúc ở `van-hanh.md`; thêm `--val-split valid` |
| Đường dẫn `D:\Python310\...` trong README cũ | Máy cũ | Dùng Python của `venv` |
