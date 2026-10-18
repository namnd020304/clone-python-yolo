# Quy trình train trên COCO 2017

Viết cho người tiếp nhận repo: chỉ đọc file này là bắt đầu train được, không cần hỏi lại. Từ 16/10/2026 dự án chỉ
train **từ số 0** trên COCO 2017 (80 class): model khởi tạo ngẫu nhiên, không nạp trọng số có sẵn, không
fine-tune, không dùng dataset biển báo nữa (QĐ-10). Bảng đầy đủ mọi tham số nằm ở
[../huong-dan/van-hanh.md](../huong-dan/van-hanh.md); lý thuyết vòng lặp train ở
[../train/05-vong-lap-train.md](../train/05-vong-lap-train.md). Cập nhật lần cuối: 16/10/2026.

Mọi lệnh viết cho PowerShell trên máy dev hiện tại (Windows 11, RTX 5070 Ti 16 GB, 28 luồng CPU), chạy từ thư mục
`clone-python-yolo/` với môi trường `.venv` đã activate (máy dev hiện còn môi trường cũ ở `..\.venv`, dùng được).

## 0. Tóm tắt một trang

```powershell
cd C:\Users\Admin\yolo-self-build\clone-python-yolo
.venv\Scripts\activate                        # môi trường cũ của máy dev: ..\.venv\Scripts\activate
python check.py                                                            # 1. phải ra 16/16
mkdir runs -Force                                                          #    thư mục chứa log, cần có trước khi dùng `> runs/...log`
python train.py --epochs 2 --eval-every 1 --out runs/thu                    # 2. chạy thử 1 phút trên coco8
python train.py --data ../data/coco --train-split train2017 --val-split val2017 --epochs 30 --batch 16 --workers 8 --lr 0.001 --eval-every 5 --out runs/coco-gpu > runs/coco-gpu.log   # 3. train thật (~8 giờ)
Get-Content runs/coco-gpu.log -Wait -Tail 5                                 # 4. xem log từ cửa sổ khác
```

Bị ngắt giữa chừng thì chạy lại đúng lệnh ở bước 3, thêm `--resume runs/coco-gpu/last.pt` và đổi `>` thành `>>`
(mục 5).

## 1. Môi trường

Môi trường Python quản lý bằng `uv`: file `pyproject.toml` và `uv.lock` nằm trong repo, môi trường ở `.venv` cạnh
chúng (Python 3.12, torch bản CUDA 12.8, cần cho RTX dòng 50).

```powershell
cd C:\Users\Admin\yolo-self-build\clone-python-yolo
uv sync                        # cài đúng phiên bản trong uv.lock (máy mới, hoặc sau khi đổi pyproject.toml)
.venv\Scripts\activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Phải in ra `True` và tên GPU. Nếu in `False` thì train sẽ chạy trên CPU (~20 lần chậm hơn): kiểm tra driver NVIDIA
(`nvidia-smi`) và kiểm tra lại rằng torch được cài từ index `pytorch-cu128` như khai báo trong `pyproject.toml`.
Đã chạy được (16/10): `torch 2.11.0+cu128`, `tqdm 4.70.1`.

Kiểm tra code trước khi train. Không qua thì **chưa train**, vì một lần train chạy nhiều giờ mà có thể ra kết quả sai:

```powershell
cd clone-python-yolo
python check.py            # phải in "Kết quả: 16/16"
```

## 2. Dữ liệu COCO 2017

### Trên máy dev: đã có sẵn

COCO đã được tải và giải nén ở `C:\Users\Admin\yolo-self-build\data\coco` (ngoài repo, không nằm trong git):

```
data/coco/
├── images/train2017/   118.287 ảnh        images/val2017/   5.000 ảnh
├── labels/train2017/   117.266 file nhãn  labels/val2017/   4.952 file nhãn
└── train2017.txt, val2017.txt, annotations/, LICENSE, README.txt
```

Ảnh không có file nhãn là ảnh không chứa vật nào, như vậy là đúng. Không có `classes.txt`, nên `train.py` tự dùng
80 tên class COCO.

Các thứ khác trong `data/`: `coco_bu/` là bản sao nhãn cũ (không có ảnh), `test2017/` không có nhãn (chỉ chấm được
qua server COCO). Cả hai **không dùng** để train. Các file `.zip` có thể xoá để lấy lại ~27 GB ổ đĩa.

### Trên máy mới: tải lại (~26 GB)

Chia và định dạng nhãn giống `coco.yaml` của Ultralytics:

```powershell
mkdir C:\Users\Admin\yolo-self-build\data; cd C:\Users\Admin\yolo-self-build\data
curl -L -o coco2017labels.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
curl -L -o train2017.zip http://images.cocodataset.org/zips/train2017.zip      # 19 GB
curl -L -o val2017.zip   http://images.cocodataset.org/zips/val2017.zip        # 0,8 GB
tar -xf coco2017labels.zip                       # tạo coco/labels/...
tar -xf train2017.zip -C coco/images
tar -xf val2017.zip   -C coco/images
```

Kiểm tra số file khớp với bảng trên:

```powershell
(Get-ChildItem coco/images/train2017).Count; (Get-ChildItem coco/labels/train2017).Count
```

**Không dùng `coco2017labels-segments.zip`.** Nhãn trong file đó là đa giác (segmentation). `YoloDataset` chỉ đọc 5
cột đầu của mỗi dòng nhãn, nên với nhãn đa giác nó sẽ đọc ra box sai mà không báo lỗi gì. Mỗi dòng nhãn đúng có
đúng 5 số: `class cx cy w h`.

## 3. Bắt đầu train

Đặt `--out` riêng cho mỗi lần chạy để không ghi đè kết quả cũ.

### 3.1. Chạy thử (bắt buộc trước một lần train dài)

```powershell
python train.py --epochs 2 --eval-every 1 --batch 4 --out runs/thu
```

Lệnh này dùng `coco8` (4 ảnh COCO, tự tải về `datasets/coco8`), mất khoảng 1 phút. Đạt khi in ra 2 dòng `epoch ...`
có `mAP50`, dòng `Xong. ...`, và `runs/thu/` có `last.pt`, `best.pt`. mAP = 0.000 ở bước này là bình thường.

### 3.2. Train COCO từ số 0 (lệnh chính)

```powershell
python train.py --data ../data/coco --train-split train2017 --val-split val2017 `
    --epochs 30 --batch 16 --workers 8 --lr 0.001 --eval-every 5 --out runs/coco-gpu > runs/coco-gpu.log
```

| Tham số | Vì sao chọn giá trị này |
|---|---|
| `--epochs 30` | ~8 giờ trên máy dev; giống job đang chạy trên GCP để so được. Muốn mAP gần bản gốc thì phải nhiều hơn hẳn (Ultralytics train 500 epoch) |
| `--batch 16` | Đo 16/10: batch 16 và 32 cùng ~136 ảnh/s, vì nút thắt là CPU đọc ảnh + mosaic chứ không phải GPU |
| `--workers 8` | Không có worker: 17 ảnh/s (~2 giờ/epoch). 8 worker: 136 ảnh/s. 16 worker cũng không nhanh hơn |
| `--lr 0.001` | lr tự tính (0.000119) là quy tắc của Ultralytics cho fine-tune / job ngắn, quá nhỏ để train từ số 0 (nhat-ky 15/10) |
| `--eval-every 5` | Mỗi lần đo mAP trên val2017 mất ~3,6 phút; đo mỗi epoch thì tốn thêm ~25% thời gian |

Thời gian đo trên máy dev (16/10): **~14,5 phút/epoch** train, cộng 3,6 phút cho mỗi lần đo mAP. Tổng 30 epoch khoảng
8 giờ. Mosaic tự tắt ở 10 epoch cuối (`--close-mosaic 10`).

Train qua đêm thì tắt chế độ ngủ của Windows trước, nếu không máy sẽ ngủ và dừng train:

```powershell
powercfg /change standby-timeout-ac 0      # sau khi train xong đặt lại, ví dụ 30 (phút)
```

### 3.3. Các tham số khác

| Tham số | Khi nào đổi | Mặc định |
|---|---|---|
| `--size` | Nhỏ hơn thì nhanh hơn nhưng kém với vật nhỏ; phải chia hết cho 32 | 640 |
| `--close-mosaic` | Tắt mosaic ở N epoch cuối. **Nếu `--epochs` ≤ N thì mosaic tắt từ epoch 1** | 10 |
| `--log-every` | In loss giữa epoch ra file log. Thanh tiến trình đã hiện sẵn trên màn hình nên thường không cần | 0 |
| `--no-augment`, `--no-ema` | Chỉ dùng cho phép thử / so sánh | tắt |
| `--weights` | **Không dùng.** Để trống = khởi tạo ngẫu nhiên, đúng mục tiêu tự build model. Truyền `yolov8n.pt` là fine-tune | trống |

Không có `--weights` thì dòng đầu log không có `Nạp ... tensor`. Thấy dòng đó là đang fine-tune, cần dừng lại kiểm tra.

## 4. Theo dõi trong lúc train

### Màn hình

Mỗi epoch có một thanh tiến trình (tqdm), cập nhật sau mỗi batch:

```
epoch 1/30:  17%|█▋        | 1254/7393 [02:47<12:01,  8.51it/s, box 3.617  cls 5.658  dfl 4.226  lr 0.00001]
```

Thanh cho biết số batch đã chạy trên tổng số batch, thời gian đã chạy và còn lại của epoch, tốc độ (batch/giây),
loss trung bình từ đầu epoch và lr. Hết epoch thì thanh biến mất, thay bằng một dòng tóm tắt (dòng này được ghi vào
log):

```
epoch    5/30  box 1.512  cls 1.893  dfl 1.402  lr 0.00097  871.2s  |  mAP50 0.xxx  mAP50-95 0.xxx
```

Các số trong dòng trên chỉ để minh hoạ định dạng. Phần `| mAP...` chỉ có ở epoch có đo mAP; khi đo có thêm một thanh
`val` (313 batch, ~3,6 phút).

### Ghi log ra file

tqdm in ra **stderr**, dòng tóm tắt in ra **stdout**. Vì vậy:

```powershell
python train.py ... > runs/coco-gpu.log          # thanh tiến trình vẫn hiện trên màn hình, file log sạch
Get-Content runs/coco-gpu.log -Wait -Tail 5      # xem log từ cửa sổ khác
nvidia-smi -l 5                                  # xem GPU có đang chạy không
```

Đừng dùng `> log 2>&1`: thanh tiến trình sẽ ghi lẫn vào file thành một dòng rất dài. Khi train bị lỗi, traceback
cũng in ra stderr, nên nó hiện trên màn hình chứ không nằm trong file log. Vì một epoch dài ~15 phút, file log sẽ
không có dòng nào trong 15 phút đầu; đó là bình thường.

### Thế nào là bình thường

| Dấu hiệu | Bình thường | Đáng lo |
|---|---|---|
| Tốc độ | ~8,5 batch/s (batch 16, 8 worker) | Dưới 2 batch/s: đang thiếu worker hoặc chạy trên CPU (dòng đầu log ghi `thiết bị cpu`) |
| Loss epoch 1 | Giảm dần từ box ~3,6, cls ~5,7, dfl ~4,3 (số đo 15/10 và 16/10) | Tăng đều, hoặc ra `nan` |
| lr | Tăng dần trong ~3 epoch đầu (warmup), rồi giảm theo cosine | — |
| mAP các epoch đầu | Thấp, tăng chậm. Train từ số 0 cần nhiều epoch | Vẫn đúng 0.000 sau 10 epoch |
| Dòng `Tắt mosaic từ epoch 21` | Xuất hiện một lần (30 − 10 + 1) | — |

Mốc so sánh (chỉ để đọc kết quả, không dùng làm điểm xuất phát): `yolov8n.pt` gốc (Ultralytics train từ số 0,
500 epoch) đo bằng code của repo trên val2017 được **mAP50-95 0.347,
mAP50 0.454** (16/10). Con số Ultralytics công bố là 0.373, cao hơn vì họ đo ảnh theo tỉ lệ gốc, còn repo này đo ảnh
vuông 640. 30 epoch từ số 0 sẽ thấp hơn mốc này nhiều.

## 5. Bị ngắt giữa chừng: resume

`last.pt` được ghi **mỗi epoch** (kèm optimizer, EMA, vị trí lr). Ghi file tạm rồi đổi tên, nên ngắt lúc đang ghi
cũng không hỏng `last.pt`. Với COCO, mỗi lần ngắt mất tối đa một epoch (~15 phút).

```powershell
python train.py --data ../data/coco --train-split train2017 --val-split val2017 `
    --epochs 30 --batch 16 --workers 8 --lr 0.001 --eval-every 5 --out runs/coco-gpu `
    --resume runs/coco-gpu/last.pt >> runs/coco-gpu.log
```

- Giữ **nguyên mọi tham số** của lệnh cũ, chỉ thêm `--resume`. Đổi `--epochs` hay `--batch` thì lịch lr bị lệch.
- Dùng `>>` để nối tiếp vào log cũ thay vì ghi đè.
- Train chạy tiếp từ epoch ngay sau epoch cuối đã lưu; epoch đang chạy dở lúc bị ngắt sẽ chạy lại từ đầu.
- Thứ tự xáo dữ liệu không giống hệt lần chạy liền một mạch, nên số đo có thể lệch nhẹ.

## 6. Báo lỗi qua WhatsApp (nên bật cho job qua đêm)

Train dừng vì lỗi thì `train.py` gửi tin nhắn gồm tên máy, epoch, lỗi và dòng code gây lỗi. Cài một lần:

```powershell
setx WHATSAPP_PHONE +84xxxxxxxxx
setx CALLMEBOT_APIKEY <apikey>     # cách lấy apikey: xem đầu file yolo/notify.py
# mở terminal mới, rồi thử:
python -m yolo.notify "thử"
```

Chưa đặt hai biến này thì chỉ in cảnh báo, train vẫn chạy bình thường. Không ghi số điện thoại hay apikey vào repo.
Nhấn Ctrl+C để dừng chủ động thì không gửi tin.

## 7. Sau khi train xong

### Kết quả nằm ở đâu

| File | Là gì | Dùng cho |
|---|---|---|
| `runs/coco-gpu/best.pt` | Trọng số EMA ở epoch có mAP50-95 cao nhất (chỉ xét epoch có đo) | **Model để dùng / bàn giao** |
| `runs/coco-gpu/last.pt` | Epoch cuối, kèm optimizer và trạng thái để resume | Train tiếp |
| `runs/coco-gpu.log` | Dòng tóm tắt từng epoch | Ghi số đo vào nhật ký |

### Kiểm tra model

```powershell
python predict.py runs/coco-gpu/best.pt ../data/coco/images/val2017/000000000139.jpg      # vẽ predict.jpg
python predict.py runs/coco-gpu/best.pt anh.jpg --conf 0.1 --out ket_qua.jpg               # hạ ngưỡng để thấy vật điểm thấp
python webcam.py --weights runs/coco-gpu/best.pt                                           # chạy với camera, chỉ hiện người
```

Đo lại mAP của một checkpoint mà không train thêm (~4 phút):

```powershell
python train.py --data ../data/coco --train-split train2017 --val-split val2017 --weights runs/coco-gpu/best.pt `
    --epochs 0 --batch 16 --workers 8 --out runs/do-lai
```

`--weights` ở đây chỉ để nạp checkpoint của chính repo vào đo, `--epochs 0` nên không train gì. Lệnh này in
`Trước khi train: mAP50 ... mAP50-95 ...` rồi dừng. Dòng `Xong. mAP50-95 tốt nhất -1.000` in ra sau đó
là bình thường; lệnh này không ghi checkpoint nào.

### Cập nhật tài liệu (bắt buộc)

1. [nhat-ky.md](nhat-ky.md): lệnh đã chạy, thời gian, mAP theo epoch, nhận xét.
2. [trang-thai.md](trang-thai.md): bảng "Kết quả đo" (kèm ngày) và mục "Model tốt nhất".
3. Model dùng để bàn giao: đưa `best.pt` và file log lên một GitHub Release mới (`*.pt` không được commit vào git),
   rồi ghi SHA-256 vào [checklist-ban-giao.md](checklist-ban-giao.md):
   `Get-FileHash runs/coco-gpu/best.pt -Algorithm SHA256`.

## 8. Job COCO trên GCP (đang chạy song song)

Từ 15/10 có một job train COCO từ số 0 chạy trên VM `yolo-coco` (`n2-standard-8`, chỉ CPU, `us-east1-b`): 30 epoch,
`--lr 0.001`, `--eval-every 1`, ~4,7 giờ/epoch, dự kiến xong ~21/10, ~56 USD credit. Máy dev chạy cùng cấu hình
nhanh hơn khoảng 20 lần (~15 phút/epoch), nên **không nên mở thêm job COCO nào trên GCP CPU**.

Lệnh dưới đây cần cài Google Cloud SDK (`gcloud`); máy dev hiện chưa có, nên phải chạy từ máy đã cài.

```powershell
$Z = "--zone=us-east1-b"
gcloud config list                                                                   # đúng tài khoản, project trước đã
gcloud compute ssh yolo-coco $Z --command="tail -5 ~/yolo/runs/coco-scratch/train.log"   # xem tiến độ

# Khi xong (VM tự tắt): lấy kết quả rồi XOÁ VM, vì ổ đĩa 150 GB vẫn tính tiền khi VM đã tắt
gcloud compute instances start yolo-coco $Z
gcloud compute ssh yolo-coco $Z --command="tar -czf runs.tgz -C ~/yolo runs"
gcloud compute scp yolo-coco:runs.tgz . $Z
tar -xzf runs.tgz
gcloud compute instances delete yolo-coco $Z --quiet
```

Job này dùng bản mosaic cũ (trước khi viết lại `load_mosaic` ngày 15/10), nên kết quả của nó và của máy dev không
hoàn toàn cùng code.

## 9. Lỗi thường gặp khi train

| Hiện tượng | Nguyên nhân | Xử lý |
|---|---|---|
| `FileNotFoundError: không có ảnh trong ...` | `--data` sai, hoặc thiếu `--train-split train2017 --val-split val2017` | Kiểm tra cấu trúc ở mục 2 |
| Dòng đầu log ghi `thiết bị cpu` | torch không thấy GPU | Xem mục 1; chạy `uv sync` lại |
| Chỉ ~1 batch/s | `--workers 0` (mặc định) | Thêm `--workers 8` |
| mAP thấp bất thường dù loss giảm, box lệch vật | Dùng nhầm nhãn đa giác (`coco2017labels-segments.zip`) | Giải nén lại `coco2017labels.zip` (mục 2) |
| `ModuleNotFoundError: tqdm` | Môi trường cài thiếu | `uv sync` |
| `RuntimeError: ... size mismatch` khi `--resume` | Đổi tham số so với lần chạy cũ | Resume phải giữ nguyên tham số |
| `CUDA out of memory` | Batch hoặc size quá lớn, hoặc chương trình khác đang chiếm GPU | Giảm `--batch`; xem `nvidia-smi` |
| Train dừng giữa đêm, không có traceback | Windows ngủ hoặc tự khởi động lại để cập nhật | `powercfg` (mục 3.2); tạm dừng Windows Update; `--resume` |
| `WinError 4551 ... Application Control policy` | Windows chặn DLL của torch (đã gặp trong terminal của trợ lý AI) | Chạy trong terminal của mình; xem `cai-dat-moi-truong.md` |

## 10. Số tham chiếu

| Thí nghiệm | Kết quả | Ngày |
|---|---|---|
| Mốc: `yolov8n.pt` gốc trên val2017 (5.000 ảnh), code của repo | mAP50 0.454, mAP50-95 0.347; đo hết 3,6 phút trên máy dev | 16/10 |
| Tốc độ train COCO, máy dev, batch 16 | 0 worker: 17 ảnh/s; 8 worker: 136 ảnh/s (~14,5 phút/epoch); 16 worker: 133 ảnh/s | 16/10 |
| Tốc độ train COCO, máy dev, batch 32, 8 worker | 137 ảnh/s, như batch 16 | 16/10 |
| Tốc độ train COCO, GCP `n2-standard-8` (CPU) | ~4,7 giờ/epoch, đo mAP val2017 ~3,6 phút | 15/10 |
| Học thuộc 4 ảnh coco8 từ số 0, 300 epoch | mAP50-95 0.995 | 11/10, 12/10 |
| COCO từ số 0, 30 epoch | Chưa có (GCP đang chạy; máy dev chưa chạy) | |

Số mới đo được thì ghi vào [trang-thai.md](trang-thai.md) trước, rồi mới chép sang bảng này.
