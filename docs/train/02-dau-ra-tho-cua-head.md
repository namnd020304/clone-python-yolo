# 02. Đầu ra thô của Head khi train

**Mục tiêu:** hiểu vì sao lúc train, Head trả **logit** chứ không trả box đã decode, và vì sao bias lớp cuối được khởi tạo
bằng những con số "lạ". Code: [yolo/head.py](../../yolo/head.py). Chấm: `python check.py 11`.

## 1. Hai chế độ của cùng một model

PyTorch có cờ `model.training`, bật bằng `model.train()`, tắt bằng `model.eval()`. Head dùng cờ này để trả hai thứ
khác nhau:

| Chế độ | Head trả về | Dùng cho |
|---|---|---|
| `model.eval()` | `(B, 84, 8400)`: cx, cy, w, h pixel + 80 điểm đã qua sigmoid | Dự đoán, đo mAP |
| `model.train()` | `box (B, 64, 8400)`, `cls (B, nc, 8400)`, `anchors (8400, 2)`, `strides (8400, 1)` | Tính loss |

Cờ này còn đổi cách BatchNorm chạy: lúc train dùng trung bình của batch hiện tại, lúc eval dùng trung bình trượt đã
tích luỹ (xem file 05).

## 2. Vì sao loss cần logit, không cần xác suất

**Logit** là số thực bất kỳ trước khi qua sigmoid; `sigmoid(logit)` mới ra xác suất 0..1.

Loss phân loại là BCE (file 04), có chứa `log(p)`. Nếu tính `p = sigmoid(x)` trước rồi mới `log(p)`:
khi `x = −20` thì `p ≈ 2e-9`, `log(p)` mất chính xác; khi `x = −120` thì `p` tròn về đúng 0 và `log(0) = −∞`.
`F.binary_cross_entropy_with_logits` gộp sigmoid và log thành một công thức ổn định, nên ta đưa logit thẳng vào đó.

Tương tự, 64 kênh box là logit của **phân bố** (4 cạnh × 16 bin). Loss DFL cần cả phân bố, không chỉ kỳ vọng của
nó (file 04).

## 3. Anchor và stride: mỗi ô biết mình ở đâu

YOLOv8 là **anchor-free**: không có "box mẫu" nào cả. Mỗi ô của lưới chỉ là một **điểm** (tâm ô), và dự đoán khoảng
cách từ điểm đó tới 4 cạnh của vật.

```python
anchors, strides = make_anchors(feats, (8, 16, 32))
# anchors (8400, 2): tâm ô, tính theo ĐƠN VỊ Ô của mức đó, ví dụ (0.5, 0.5), (1.5, 0.5), ...
# strides (8400, 1): 8 cho 6400 ô đầu, 16 cho 1600 ô tiếp, 32 cho 400 ô cuối
```

Đổi đơn vị: `pixel = đơn vị ô × stride`. Ví dụ ô (10, 4) ở mức stride 16 có tâm tại
`((10 + 0.5) × 16, (4 + 0.5) × 16) = (168, 72)` pixel.

Ba mức lo ba cỡ vật: mức stride 8 (lưới mịn 80×80) bắt vật nhỏ, mức stride 32 (lưới thô 20×20) bắt vật lớn. DFL dự
đoán khoảng cách tối đa 15 ô, tức tối đa 15 × 32 = 480 pixel mỗi phía ở mức thô nhất.

## 4. Khởi tạo bias: bắt đầu từ một dự đoán hợp lý

`init_bias()` đặt bias lớp cuối:

- **Nhánh box:** bias = 2.0, đều cho mọi bin.
- **Nhánh class:**

$$
b = \log\frac{5}{n_c \cdot (640 / s)^2}
$$

Ý nghĩa: giả sử trung bình mỗi ảnh có khoảng 5 vật, rải đều trên `nc` class và trên `(640/s)²` ô của mức đó. Xác suất
một ô bất kỳ chứa một class cụ thể vào khoảng `5 / (nc × số ô)`. Đặt bias sao cho `sigmoid(b)` ≈ đúng con số đó:

| nc | stride | số ô | bias | sigmoid(bias) |
|---|---|---|---|---|
| 80 | 8 | 6400 | −11.54 | 9.8e-6 |
| 80 | 32 | 400 | −8.76 | 1.6e-4 |
| 4 | 8 | 6400 | −8.54 | 2.0e-4 |
| 4 | 32 | 400 | −5.77 | 3.1e-3 |

Vì sao quan trọng? Phần lớn trong 8400 ô là **nền** (không có vật). Nếu bias = 0 thì ban đầu mọi ô đoán xác suất 0.5
cho mọi class: hàng trăm nghìn dự đoán sai, loss khổng lồ, gradient đầu tiên đẩy model lung tung. Bắt đầu từ "gần như
không có gì" thì model chỉ cần học chỗ nào *có* vật.

Bias này chỉ có tác dụng khi train từ đầu. Khi nạp `yolov8n.pt`, trọng số đã học sẽ đè lên.

## 5. Vì sao nhánh class "không nạp được" khi đổi số class

Khi fine-tune với 4 class, log ghi `Nạp 319/355 tensor`. Lý do: số kênh nhánh class là
`max(64, min(nc, 100))`. Với 80 class là 80 kênh, với 4 class là 64 kênh. Shape khác nên cả 3 nhánh class (mỗi nhánh 2
Conv + 1 Conv2d) được học lại từ đầu. Backbone, neck và nhánh box vẫn dùng trọng số gốc, đó là phần quý nhất: chúng đã
biết "nhìn" cạnh, góc, hình dạng.

## Tự kiểm tra

1. Ô thứ 6400 (đếm từ 0) thuộc mức nào, tâm ở pixel nào? *(Đáp án: ô đầu tiên của mức stride 16, tâm (8, 8).)*
2. Nếu dataset có trung bình 50 vật mỗi ảnh (ví dụ đếm người trong đám đông), bias class nên lớn hơn hay nhỏ hơn?
3. Vì sao không dùng `F.binary_cross_entropy(torch.sigmoid(x), y)`?
