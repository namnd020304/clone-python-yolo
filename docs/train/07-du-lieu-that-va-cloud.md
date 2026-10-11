# 07. Dữ liệu thật và train trên cloud

**Mục tiêu:** đi từ "code chạy đúng" đến "model dùng được": lấy dataset thật, chuyển định dạng, train lâu trên máy
mạnh hơn. Code: [kaggle_data.py](../../kaggle_data.py), [train.py](../../train.py), [predict.py](../../predict.py).

## 1. Quy trình đầy đủ

```
Kaggle (owner/ten-dataset)
   │  kaggle_data.py: tải, nhận định dạng, đổi sang YOLO, chia train/val, ghi classes.txt
   ▼
datasets/<tên>/images/{train,val}, labels/{train,val}, classes.txt
   │  train.py --weights yolov8n.pt  (máy nhà hoặc máy ảo cloud)
   ▼
runs/<tên>/best.pt, last.pt, train.log
   │  predict.py
   ▼
ảnh có box
```

## 2. Hai định dạng nhãn phổ biến

| | YOLO | Pascal VOC |
|---|---|---|
| File | `.txt` mỗi ảnh | `.xml` mỗi ảnh |
| Box | `class cx cy w h`, chia cho kích thước ảnh | `<xmin> <ymin> <xmax> <ymax>` pixel |
| Class | số nguyên; tên ở `data.yaml` / `classes.txt` | tên chữ `<name>` |

Đổi VOC sang YOLO (`W, H` là kích thước ảnh, đọc trong thẻ `<size>`):

$$
c_x = \frac{x_{\min} + x_{\max}}{2W}, \quad c_y = \frac{y_{\min} + y_{\max}}{2H}, \quad
w = \frac{x_{\max} - x_{\min}}{W}, \quad h = \frac{y_{\max} - y_{\min}}{H}
$$

**Ví dụ thật** (`road0.png` của dataset biển báo, ảnh rộng 267, cao 400): đèn giao thông `xmin 98, xmax 208, ymin 62,
ymax 232` → `cx = 306 / 534 = 0.573`, `cy = 294 / 800 = 0.3675`, `w = 110 / 267 = 0.412`, `h = 170 / 400 = 0.425`.
Đúng với dòng `3 0.573034 0.367500 0.411985 0.425000` script ghi ra.

Tên class VOC được xếp theo ABC rồi đánh số: `0=crosswalk, 1=speedlimit, 2=stop, 3=trafficlight`.

Các định dạng khác (COCO JSON, CSV) và nhãn segmentation dạng đa giác chưa được hỗ trợ; script sẽ báo không nhận ra.

## 3. Chia train / val

- Dataset đã có thư mục `train`, `val` (hoặc `valid`), `test`: giữ nguyên. `test` chỉ dùng làm val khi không có val.
- Chưa chia: trộn ngẫu nhiên (seed 0, nên lần nào chạy cũng ra cùng một cách chia), 20% làm val.

Vì sao phải tách val? Đo trên ảnh đã train thì chỉ biết model nhớ tốt không, không biết nó có dùng được với ảnh mới
không. Ảnh val **không bao giờ** được dùng để train.

Một bẫy hay gặp: ảnh gần trùng nhau, ví dụ các khung hình liên tiếp của một video, bị chia một nửa vào train, một nửa
vào val. mAP val khi đó cao giả. Với dataset từ video, nên chia theo video chứ không chia theo ảnh.

## 4. Train ở đâu: CPU, GPU, cloud

Thời gian đo thật, dataset biển báo 702 ảnh, ảnh 640:

| Máy | Thời gian mỗi epoch |
|---|---|
| Laptop (CPU) | ~140–185 giây, tính cả lần đo mAP |
| Máy ảo GCP `n2-standard-8` (8 vCPU) | ~100 giây |
| GPU (T4, L4) | chưa đo; thường nhanh hơn CPU nhiều lần |

Cách giảm thời gian: `--size 416` (ít điểm ảnh hơn khoảng 2,4 lần, mất một chút độ chính xác với vật nhỏ), hoặc tăng
`--eval-every` để đo mAP ít lần hơn.

## 5. Train qua đêm trên máy ảo GCP

Tài khoản dùng thử của GCP không thuê được GPU, nên dùng máy CPU nhiều nhân. Các bước đã làm:

1. **Tạo VM**: `n2-standard-8`, Debian 12, ổ 30 GB, vùng `us-central1-a` (rẻ hơn vùng châu Á).
2. **Gửi code** (không gửi `venv`, `datasets`). Trọng số `yolov8n.pt` được chuyển trước sang định dạng của repo
   (`yolov8n_mine.pt`), nên VM không cần cài `ultralytics`.
3. **Cài môi trường**: `python3 -m venv`, torch bản CPU, rồi `numpy pillow pyyaml kagglehub`. Nhớ nâng `pip` trước:
   `pip` sẵn có của Debian quá cũ, cài torch bị lỗi.
4. **Tải dataset ngay trên VM** bằng `kaggle_data.py`. Mạng của Google tải 218 MB trong 2 giây.
5. **Chạy `check.py`** trên VM trước khi train lâu, để chắc môi trường mới chạy đúng.
6. **Chạy train trong `tmux`**: SSH ngắt thì train vẫn chạy. Ghi log ra file bằng `tee`, dùng `python -u` để log ghi
   ngay chứ không bị đệm.
7. **Tự tắt khi xong**: `sudo shutdown -h +2` ở cuối script. VM tắt thì hết tính tiền CPU, ổ đĩa vẫn giữ kết quả.
8. **Hôm sau**: bật VM, nén `runs/`, `gcloud compute scp` về máy, rồi **xoá VM** (ổ đĩa vẫn bị tính khoảng 3
   USD/tháng nếu để đó).

Chi phí ước tính: khoảng 0.39 USD/giờ × 3–4 giờ ≈ 1,5 USD, trừ vào credit dùng thử.

## 6. Khi kết quả chưa tốt: kiểm tra theo thứ tự

1. **Nhãn có đúng không?** Vẽ nhãn lên vài ảnh và nhìn. Lỗi dữ liệu phổ biến hơn lỗi code nhiều.
2. **Loss train có giảm không?** Không giảm thì xem lại lr, hoặc chạy phép thử học thuộc (file 05).
3. **Loss train giảm, mAP val không tăng?** Overfit: cần thêm dữ liệu, augmentation mạnh hơn, hoặc dừng sớm hơn.
4. **mAP50 cao, mAP50-95 thấp?** Tìm được vật nhưng box chưa khít: train lâu hơn, ảnh lớn hơn.
5. **Một class kém hẳn?** Thường do class đó quá ít ảnh. Đếm số box của từng class trong tập train.

## Tự kiểm tra

1. Đổi box VOC `xmin 0, ymin 0, xmax 100, ymax 50` trên ảnh 200 × 100 (rộng × cao) sang YOLO.
   *(Đáp án: `cx 0.25, cy 0.25, w 0.5, h 0.5`.)*
2. Vì sao chạy `kaggle_data.py` ngay trên VM tốt hơn tải về máy nhà rồi gửi lên?
3. Train xong mà quên xoá VM 1 tháng thì tốn những khoản gì?
