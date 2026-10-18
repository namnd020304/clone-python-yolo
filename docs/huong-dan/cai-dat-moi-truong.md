# Cài đặt môi trường

Mục tiêu: từ máy trắng đến khi `python check.py` báo **Kết quả: 16/16** và torch thấy GPU. Viết cho máy dev hiện tại:
Windows 11, GPU NVIDIA RTX 5070 Ti 16 GB. Cập nhật lần cuối: 16/10/2026.

## 1. Công cụ cần cài

| Công cụ | Phiên bản đã dùng | Ghi chú |
|---|---|---|
| Driver NVIDIA | — | `nvidia-smi` phải chạy được và hiện tên GPU |
| `uv` | — | Quản lý Python và thư viện. Cài: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"`; nằm ở `C:\Users\<bạn>\.local\bin\uv.exe` |
| git | — | |
| Camera | — | Chỉ cần cho `webcam.py` |
| Ổ đĩa trống | ~50 GB | COCO 2017 giải nén ~20 GB, file zip ~27 GB (xoá được sau khi giải nén) |

Không cần cài Python riêng: `uv` tự tải Python 3.12 theo `pyproject.toml`. Chưa có `uv` trong PATH thì thêm
`C:\Users\<bạn>\.local\bin` vào PATH (`yolo-self-build/.vscode/settings.json` đã đặt sẵn cho terminal VS Code).

## 2. Cấu trúc thư mục

`pyproject.toml` và `uv.lock` nằm **trong repo** (từ 16/10), nên `git clone` là có đủ cấu hình môi trường. Dữ liệu COCO
nằm ngoài repo, ở thư mục cha, vì quá lớn:

```
C:\Users\Admin\yolo-self-build\
├── data\coco\                   COCO 2017 (xem ../ban-giao/quy-trinh-train.md mục 2); lệnh train trỏ tới ../data/coco
└── clone-python-yolo\           repo git này; mọi lệnh python chạy từ đây
    ├── pyproject.toml, uv.lock  danh sách thư viện và phiên bản khoá cứng
    └── .venv\                   môi trường Python (uv tạo, đã gitignore)
```

Riêng máy dev hiện tại còn một môi trường cũ ở `yolo-self-build\.venv` (tạo trước khi hai file này vào repo). Dùng
được nguyên trạng bằng `..\.venv\Scripts\activate`; muốn gọn thì chạy `uv sync` trong repo để có `.venv` riêng rồi xoá
thư mục cũ.

## 3. Lấy code và cài thư viện

```powershell
mkdir C:\Users\Admin\yolo-self-build; cd C:\Users\Admin\yolo-self-build
git clone https://github.com/namnd020304/clone-python-yolo.git
cd clone-python-yolo
uv sync
.venv\Scripts\activate
```

`uv sync` cài đúng phiên bản trong `uv.lock`. torch và torchvision lấy từ index `pytorch-cu128` (bản CUDA 12.8),
vì GPU dòng RTX 50 (kiến trúc Blackwell) cần CUDA ≥ 12.8. Thêm hoặc đổi thư viện thì sửa `pyproject.toml` rồi chạy
lại `uv sync`.

Phiên bản đã chạy được (16/10/2026): Python 3.12, `torch 2.11.0+cu128`, `tqdm 4.70.1`.

| Thư viện | Dùng cho |
|---|---|
| `torch`, `torchvision`, `numpy`, `pillow`, `tqdm` | Bắt buộc: model, train, `check.py` |
| `ultralytics` | Chỉ để đối chiếu: `load_weights.py`, `check.py` bài 12–13 |
| `opencv-python` | `webcam.py` |
| `pyyaml`, `kagglehub` | `kaggle_data.py` (không còn dùng trong hướng train hiện tại, QĐ-10) |

Các lệnh trong tài liệu viết là `python ...` với `.venv` đã activate. Không activate thì gọi thẳng
`.venv\Scripts\python.exe ...` từ thư mục repo (máy dev có môi trường cũ: `..\.venv\Scripts\python.exe`).

## 4. Trọng số và dữ liệu

| Thứ | Lấy thế nào | Dùng cho |
|---|---|---|
| `data/coco` | Đã có trên máy dev; máy mới tải theo [../ban-giao/quy-trinh-train.md](../ban-giao/quy-trinh-train.md) mục 2 | **Train** |
| `datasets/coco8` | Tự tải khi chạy `train.py` với `--data coco8` (mặc định) | Chạy thử nhanh |
| `yolov8n.pt` | Tự tải khi chạy `load_weights.py`, hoặc `curl -L -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt` | Chỉ đối chiếu và làm mốc mAP, **không** dùng làm điểm xuất phát train (QĐ-10) |
| `yolov8n_mine.pt` | Đổi `yolov8n.pt` sang định dạng của repo (lệnh bên dưới) | `webcam.py` mặc định, khi chưa có model tự train |

```powershell
python -c "import torch; from train import load_pretrained; from yolo.model import YOLOv8n; from yolo.dataset import COCO_NAMES; m = YOLOv8n(); load_pretrained(m, 'yolov8n.pt'); torch.save({'model': m.state_dict(), 'nc': 80, 'names': COCO_NAMES}, 'yolov8n_mine.pt')"
```

Tất cả nằm ngoài git (`.gitignore` chặn `*.pt`, `datasets/`, `runs/`; `data/` nằm ngoài repo).

## 5. Kiểm tra

```powershell
nvidia-smi                                                                       # thấy GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # phải có True
python check.py            # 16 bài; bài 12-13 cần ultralytics
python check.py 7          # chỉ chấm một bài
python load_weights.py     # so với Ultralytics, ra result.jpg: 4 người + 1 xe buýt, lệch ~6e-05
python train.py --epochs 2 --eval-every 1 --out runs/thu    # chạy thử train trên coco8; dòng đầu ghi "thiết bị cuda"
```

## 6. Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `torch.cuda.is_available()` ra `False` | Driver NVIDIA cũ / thiếu, hoặc torch bản CPU | `nvidia-smi`; chạy lại `uv sync` để lấy torch từ `pytorch-cu128` |
| `CUDA error: no kernel image is available` | torch build cho CUDA cũ hơn 12.8, không hỗ trợ RTX 50 | Giữ index `pytorch-cu128` trong `pyproject.toml`, `uv sync` |
| `uv: command not found` | `C:\Users\<bạn>\.local\bin` chưa nằm trong PATH | Thêm vào PATH, hoặc gọi đường dẫn đầy đủ `uv.exe` |
| `OSError: [WinError 4551] An Application Control policy has blocked this file ... torch\lib\*.dll` | Windows (Smart App Control / chính sách ứng dụng) chặn DLL của torch. Đã gặp trong terminal của trợ lý AI (11/10, 15/10) | Chạy lệnh trong terminal VS Code / PowerShell của mình. Nếu vẫn bị chặn: kiểm tra Smart App Control trong Windows Security |
| `UnicodeEncodeError: 'charmap' ...` khi in tiếng Việt | Console Windows dùng bảng mã cũ | `check.py`, `train.py` đã tự chuyển UTF-8. Script khác: đặt `PYTHONUTF8=1` |
| `ModuleNotFoundError: ...` | Chưa activate `.venv`, hoặc chưa `uv sync` | `.venv\Scripts\activate`, `uv sync` |
| `FileNotFoundError: không có ảnh trong ...` | `--data` hoặc `--train-split` / `--val-split` sai | Xem `../ban-giao/quy-trinh-train.md` mục 2 |
