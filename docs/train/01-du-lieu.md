# 01. Dữ liệu: từ file ảnh đến tensor

**Mục tiêu:** hiểu nhãn được lưu thế nào, và biến đổi ra sao để khớp với ảnh sau letterbox. Code:
[yolo/dataset.py](../../yolo/dataset.py). Chấm: `python check.py 10`.

## 1. Định dạng nhãn YOLO

Mỗi ảnh có một file `.txt` cùng tên. Mỗi dòng là một vật:

```
class  cx      cy      w       h
3      0.5730  0.3675  0.4120  0.4250
```

- `class`: số nguyên `0 … nc−1`. Tên class nằm ở file riêng (`classes.txt`, dòng thứ i là class i).
- `cx, cy`: **tâm** box. `w, h`: rộng, cao. Cả 4 số đều **chia cho kích thước ảnh**, nên nằm trong 0..1.

Vì sao chia cho kích thước ảnh? Để nhãn không phụ thuộc độ phân giải: resize ảnh bao nhiêu thì nhãn vẫn đúng.

Thư mục theo quy ước "đổi `images` thành `labels`":

```
datasets/bien-bao/images/train/road0.png
datasets/bien-bao/labels/train/road0.txt
```

## 2. Đổi nhãn theo letterbox

Model chỉ nhận ảnh 640×640. Letterbox (bước 8) co ảnh theo hệ số `scale`, rồi thêm viền `pad_x`, `pad_y`. Box phải
chịu đúng phép biến đổi đó:

$$
x_{640} = x_{\text{gốc}} \cdot \text{scale} + \text{pad}_x \qquad y_{640} = y_{\text{gốc}} \cdot \text{scale} + \text{pad}_y
$$

Quy trình trong `__getitem__`:

1. `cx, cy, w, h` (0..1) → nhân rộng, cao ảnh gốc → `x1, y1, x2, y2` theo pixel ảnh gốc.
2. Nhân `scale`, cộng `pad` → pixel trong ảnh 640.

**Ví dụ** (đúng bài trong `check.py 10`): ảnh 480 cao × 640 rộng, nhãn `3 0.5 0.5 0.5 0.5`.

| Bước | x1 | y1 | x2 | y2 |
|---|---|---|---|---|
| Pixel ảnh gốc | 0.25 × 640 = 160 | 0.25 × 480 = 120 | 480 | 360 |
| `scale = 640 / 640 = 1`, `pad = (0, 80)` | 160 | 120 + 80 = **200** | 480 | 360 + 80 = **440** |

Từ đây trở đi, mọi box thật (gọi là **gt**, ground truth) đều ở dạng `x1, y1, x2, y2` pixel ảnh 640.

## 3. Augmentation: tạo thêm ảnh "mới" từ ảnh cũ

Với ít dữ liệu, model dễ **học thuộc** (overfit): nhớ đúng từng ảnh train thay vì học đặc điểm chung. Augmentation
biến đổi ảnh ngẫu nhiên mỗi lần đọc, nên model không bao giờ thấy hai lần cùng một ảnh y hệt.

Repo này làm ba phép (chỉ khi train, `augment=True`):

| Phép | Ảnh | Nhãn |
|---|---|---|
| Mosaic, xác suất `--mosaic` (mặc định 1.0) | Ghép 4 ảnh, co giãn ×0.5–1.5, cắt về 640 | Dịch theo chỗ dán, nhân hệ số co giãn, cắt theo khung |
| Lật ngang, xác suất 0.5 | `x.flip(-1)` | `x1_mới = 640 − x2_cũ`, `x2_mới = 640 − x1_cũ` |
| Đổi độ sáng ×0.75 … ×1.25 | nhân rồi kẹp về 0..1 | không đổi |

Để ý cách đổi nhãn khi lật: cạnh phải cũ thành cạnh trái mới, nên phải **đổi chỗ** x1 và x2. Nếu chỉ lấy
`640 − x1` thì được box có x1 > x2, tức box âm.

### Mosaic

Hàm `load_mosaic` trong [yolo/dataset.py](../../yolo/dataset.py):

```
khung 1280 x 1280, nền xám 114
┌─────────┬─────────┐
│  ảnh i  │ ngẫu nhiên│     4 ảnh (mỗi ảnh co về cạnh dài 640) chạm góc vào tâm ●,
├─────────●─────────┤     tâm chọn ngẫu nhiên trong [320, 960]
│ngẫu nhiên│ngẫu nhiên│
└─────────┴─────────┘
→ lấy cửa sổ cỡ 640 / r quanh giữa khung (r ngẫu nhiên 0.5–1.5, dịch ±10%), co về 640 x 640
```

- **r < 1** (cửa sổ to hơn 640): mọi vật bị thu nhỏ. Model được luyện nhiều với **vật nhỏ**, đúng điểm yếu đã thấy ở
  dataset biển báo (bỏ sót biển 7 × 9 px).
- **r > 1**: vật được phóng to, có khi bị cắt mất một phần.
- Một ảnh chứa vật của 4 ảnh, với nền và bối cảnh lạ: model khó học thuộc hơn.

Nhãn phải đi qua đúng các phép đó: cộng độ dời chỗ dán, cắt theo phần ảnh thực sự được dán, trừ góc cửa sổ, nhân r,
cắt về [0, 640]. Box bị cắt gần hết thì bỏ: cạnh còn < 2 px, hoặc còn < 10% diện tích.

Ảnh mosaic không giống ảnh thật lúc dùng. Vì vậy `train.py` **tắt mosaic ở 10 epoch cuối** (`--close-mosaic 10`) để
model quen lại với ảnh letterbox bình thường. Ultralytics cũng làm vậy.

Ultralytics còn dùng thêm xoay, đổi màu HSV, mixup. Đó là chỗ có thể làm tiếp.

Phép biến đổi nào cũng phải biến đổi nhãn theo đúng cách. Phép nào chỉ đổi màu thì nhãn giữ nguyên; phép nào đổi hình
học (lật, co, xoay, cắt) thì nhãn phải đổi theo.

## 4. Gom batch: mỗi ảnh một số box khác nhau

Ảnh thì ghép được ngay: cùng shape `(3, 640, 640)` → `(B, 3, 640, 640)`. Nhãn thì không: ảnh 1 có 3 vật, ảnh 2 có 1
vật. Tensor phải là hình chữ nhật, nên ta **đệm** cho đủ `M` = số vật nhiều nhất trong batch, và kèm một **mask** đánh
dấu dòng nào thật:

```
gt_bboxes (B=2, M=3, 4)          mask_gt (2, 3, 1)
ảnh 0: [box, box, box]          [1, 1, 1]
ảnh 1: [box, 0,   0  ]          [1, 0, 0]   ← hai dòng đệm
```

Mọi bước sau (assigner, loss) phải nhân với `mask_gt` để dòng đệm không ảnh hưởng gì. Ảnh không có vật nào cũng hợp
lệ: nó dạy model rằng "ở đây không có gì".

## Tự kiểm tra

1. Ảnh 720 × 1280 (cao × rộng), nhãn `0 0.5 0.5 0.25 0.5`. Sau letterbox (`scale = 0.5`, `pad = (0, 140)`), box là
   bao nhiêu? *(Đáp án: x1 = 0.375 × 1280 × 0.5 = 240, y1 = 0.25 × 720 × 0.5 + 140 = 230, x2 = 400, y2 = 410.)*
2. Nếu lật ngang mà quên đổi nhãn, model sẽ học sai thế nào?
3. Vì sao không ghép nhãn bằng `torch.stack` được, còn ảnh thì được?
