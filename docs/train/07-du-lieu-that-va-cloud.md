# 07. Dữ liệu thật: COCO 2017 và train trên GPU

**Mục tiêu:** đi từ "code chạy đúng trên 4 ảnh" đến "model tự train từ số 0 dùng được": hiểu bộ dữ liệu COCO 2017,
định dạng nhãn, và vì sao tốc độ train phụ thuộc vào CPU nhiều hơn GPU. Code: [train.py](../../train.py),
[yolo/dataset.py](../../yolo/dataset.py). Lệnh chạy cụ thể: [../ban-giao/quy-trinh-train.md](../ban-giao/quy-trinh-train.md).

Từ 16/10/2026 repo chỉ train từ số 0 trên COCO (QĐ-10). Dataset biển báo và `kaggle_data.py` của các buổi trước vẫn
còn trong repo nhưng không dùng nữa; số đo cũ nằm trong [../ban-giao/nhat-ky.md](../ban-giao/nhat-ky.md).

## 1. Quy trình đầy đủ

```
COCO 2017: ảnh (images.cocodataset.org) + nhãn đã đổi sang YOLO (coco2017labels.zip của Ultralytics)
   │  giải nén
   ▼
data/coco/images/{train2017,val2017}, labels/{train2017,val2017}
   │  train.py, KHÔNG có --weights: trọng số ngẫu nhiên (GPU máy dev)
   ▼
runs/coco-gpu/best.pt, last.pt + runs/coco-gpu.log
   │  predict.py / webcam.py
   ▼
ảnh có box
```

## 2. COCO 2017 là gì

| Phần | Số ảnh | Có nhãn | Dùng làm gì |
|---|---|---|---|
| `train2017` | 118.287 | 117.266 file (1.021 ảnh không có vật nào) | Train |
| `val2017` | 5.000 | 4.952 file | Đo mAP. **Không bao giờ** dùng để train |
| `test2017` | 40.670 | Không | Chỉ chấm được bằng cách gửi kết quả lên server COCO; repo không dùng |

80 class đồ vật thường gặp: người, xe, con vật, đồ dùng trong nhà... Mỗi ảnh trung bình ~7 vật (val2017: 36.335 box / 5.000 ảnh), nhiều vật rất nhỏ.
Đây chính là bộ dữ liệu Ultralytics dùng để train `yolov8n.pt`, nên mAP của model tự train so được trực tiếp với bản
gốc.

Vì sao phải tách val? Đo trên ảnh đã train thì chỉ biết model nhớ tốt không, không biết nó có dùng được với ảnh mới
không. Phép thử học thuộc 4 ảnh (file 05) cố ý làm ngược lại: train và đo trên cùng tập, để kiểm tra code.

## 3. Hai định dạng nhãn: COCO JSON và YOLO

COCO gốc ghi mọi nhãn vào một file JSON lớn (`annotations/instances_val2017.json`). Repo đọc định dạng YOLO: mỗi ảnh
một file `.txt`. `coco2017labels.zip` của Ultralytics là bản đã đổi sẵn.

| | COCO JSON | YOLO `.txt` |
|---|---|---|
| File | Một file cho cả tập | Một file mỗi ảnh |
| Box | `[x, y, w, h]` pixel, `(x, y)` là góc **trên trái** | `class cx cy w h`, `(cx, cy)` là **tâm**, chia cho kích thước ảnh |
| Class | `category_id` từ 1 đến 90, có chỗ trống (chỉ 80 số được dùng) | Số liên tục 0 … 79 |

Đổi từ COCO sang YOLO (`W, H` là kích thước ảnh):

$$
c_x = \frac{x + w/2}{W}, \quad c_y = \frac{y + h/2}{H}, \quad w' = \frac{w}{W}, \quad h' = \frac{h}{H}
$$

**Ví dụ thật** (ảnh `val2017/000000000139.jpg`, rộng 640, cao 426): trong JSON có chậu cây `category_id 64`, bbox
`[237.0, 142.5, 24.7, 69.5]`.

- `cx = (237.0 + 12.35) / 640 = 0.3896`, `cy = (142.5 + 34.75) / 426 = 0.4161`
- `w = 24.7 / 640 = 0.0386`, `h = 69.5 / 426 = 0.1631`
- `category_id 64` là class thứ 59 trong danh sách 80 class, tức số **58** (đếm từ 0): `potted plant`.

Khớp với dòng `58 0.389578 0.416103 0.0385937 0.163146` trong `labels/val2017/000000000139.txt`. Chậu cây này chỉ
rộng 25 pixel: vật nhỏ như thế rất phổ biến trong COCO.

**Bẫy:** Ultralytics còn có `coco2017labels-segments.zip`, nhãn dạng đa giác (`class x1 y1 x2 y2 ...`). `YoloDataset`
chỉ lấy 5 cột đầu, nên với file đó nó đọc ra box sai mà **không báo lỗi**. Kiểm tra: mỗi dòng nhãn có đúng 5 số.

## 4. Train ở đâu: CPU, GPU, cloud

Thời gian đo thật, COCO train2017, ảnh 640:

| Máy | Mỗi epoch | 30 epoch | Ngày đo |
|---|---|---|---|
| Máy ảo GCP `n2-standard-8` (8 vCPU, không GPU) | ~4,7 giờ | ~6 ngày | 15/10 |
| Máy dev RTX 5070 Ti, `--workers 0` | ~2 giờ | ~2,5 ngày | 16/10 |
| Máy dev RTX 5070 Ti, `--workers 8` | **~14,5 phút** | ~8 giờ (kể cả đo mAP mỗi 5 epoch) | 16/10 |

Có GPU vẫn chậm nếu không có worker. Mỗi ảnh train phải được đọc từ đĩa, giải mã JPEG, ghép mosaic từ 4 ảnh, lật,
đổi độ sáng. Phần này chạy trên **CPU**. Với `--workers 0`, GPU ngồi đợi CPU chuẩn bị từng batch. Với 8 tiến trình
worker chuẩn bị batch song song, tốc độ lên ~136 ảnh/s.

Tăng thêm nữa không giúp: 16 worker cũng ~133 ảnh/s; batch 32 cũng ~137 ảnh/s như batch 16. Vậy nút thắt nhiều khả năng
không còn ở đọc ảnh hay ở GPU mà ở tiến trình chính (chưa đo riêng từng phần) (gom batch, gán nhãn TAL, cập nhật EMA). Muốn nhanh hơn nữa thì
phải tối ưu những phần đó, ví dụ chạy TAL hoàn toàn trên GPU, hoặc dùng AMP (số thực 16-bit).

Đo mAP trên cả 5.000 ảnh val mất ~3,6 phút. Phần chậm là NMS và ghép dự đoán với nhãn, chạy trên CPU. Vì vậy đo mỗi 5
epoch thay vì mỗi epoch.

## 5. Train qua đêm trên máy nhà

1. **Chạy thử trước** trên coco8 (1 phút) để chắc môi trường đúng.
2. **Tắt chế độ ngủ**: `powercfg /change standby-timeout-ac 0`. Máy ngủ là train dừng.
3. **Bật báo lỗi WhatsApp** (`yolo/notify.py`): train chết lúc 3 giờ sáng thì sáng ra biết ngay.
4. **Ghi log ra file** bằng `> runs/<ten>.log`. Thanh tiến trình in ra stderr nên vẫn hiện trên màn hình.
5. **Bị ngắt** (mất điện, Windows Update tự khởi động lại): chạy lại đúng lệnh, thêm `--resume runs/<ten>/last.pt`.
   Mất tối đa một epoch.

Máy ảo GCP chỉ CPU (cách làm trước 16/10, QĐ-08) chậm hơn máy dev ~20 lần với COCO, nên không dùng nữa.

## 6. Khi kết quả chưa tốt: kiểm tra theo thứ tự

1. **Nhãn có đúng không?** Vẽ nhãn lên vài ảnh và nhìn; kiểm tra mỗi dòng nhãn có đúng 5 số. Lỗi dữ liệu phổ biến
   hơn lỗi code nhiều.
2. **Có đúng là train từ số 0 không?** Dòng đầu log không được có `Nạp ... tensor`.
3. **Loss train có giảm không?** Không giảm thì xem lại lr, hoặc chạy phép thử học thuộc (file 05).
4. **Loss giảm, mAP val tăng rất chậm?** Bình thường với train từ số 0: Ultralytics cần 500 epoch để đạt mAP50-95
   0.373. 30 epoch chỉ là bước đầu.
5. **mAP50 cao, mAP50-95 thấp?** Tìm được vật nhưng box chưa khít: train lâu hơn.
6. **Một class kém hẳn?** Thường do class đó ít ảnh (COCO lệch mạnh: trên val2017, `person` chiếm 30% số box).

## Tự kiểm tra

1. Đổi box COCO `[0, 0, 100, 50]` trên ảnh 200 × 100 (rộng × cao) sang YOLO.
   *(Đáp án: `cx 0.25, cy 0.25, w 0.5, h 0.5`.)*
2. GPU mạnh gấp nhiều lần CPU, vậy sao `--workers 0` trên máy có GPU vẫn mất ~2 giờ/epoch?
3. `category_id` của COCO đi từ 1 đến 90 nhưng chỉ có 80 class. Nếu đổi sang YOLO bằng `class = category_id − 1` thì
   sai ở đâu?
