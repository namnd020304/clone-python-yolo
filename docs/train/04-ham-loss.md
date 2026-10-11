# 04. Hàm loss: đo model sai bao nhiêu

**Mục tiêu:** hiểu ba thành phần loss của YOLOv8 và vì sao mỗi cái có mặt. Code: [yolo/loss.py](../../yolo/loss.py),
`bbox_ciou` trong [yolo/postprocess.py](../../yolo/postprocess.py). Chấm: `python check.py 13` (so với Ultralytics).

$$
\mathcal{L} = 7.5 \cdot \mathcal{L}_{\text{box}} + 0.5 \cdot \mathcal{L}_{\text{cls}} + 1.5 \cdot \mathcal{L}_{\text{dfl}}
$$

## 0. Các bước trong `DetectionLoss.__call__`

1. Đổi shape: `box (B, 64, A)` → `pred_dist (B, A, 4, 16)`; `cls (B, nc, A)` → `pred_scores (B, A, nc)`.
2. **Decode DFL** ra box dự đoán theo đơn vị ô: `dist = softmax(pred_dist) @ [0, 1, …, 15]`, rồi
   `x1y1 = anchor − dist[:2]`, `x2y2 = anchor + dist[2:]`.
3. Gọi **TAL** (file 03), đưa vào phiên bản đã `detach` và đổi ra pixel. Nhận về `target_bboxes`, `target_scores`,
   `fg_mask`.
4. Tính 3 loss, chia cho `ts_sum = target_scores.sum()`, nhân hệ số.

## 1. Loss phân loại: BCE

Mỗi ô, mỗi class là một câu hỏi có/không độc lập. Đáp án là `target_scores`: 0 ở ô nền, nhãn mềm (≤ 1) ở ô dương.
Binary Cross-Entropy với đích `t` và xác suất `p = sigmoid(x)`:

$$
\text{BCE}(x, t) = -\left[ t \log p + (1 - t) \log (1 - p) \right]
$$

- Tính trên **mọi** ô và mọi class: 8400 × nc số hạng. Ô nền cũng học, cụ thể là học nói "không".
- Dùng sigmoid độc lập cho từng class, không dùng softmax giữa các class. Nhờ vậy một vật có thể vừa là "người" vừa là
  "người đi xe đạp" nếu dataset gán vậy.
- Cộng tất cả rồi chia `ts_sum`. Chia cho tổng nhãn mềm (xấp xỉ số ô dương có trọng số) chứ không chia cho 8400 × nc,
  để loss không bị pha loãng bởi biển ô nền.

## 2. Loss box: CIoU

### Từ IoU đến CIoU

IoU = diện tích giao / diện tích hợp. Dùng `1 − IoU` làm loss có một lỗ hổng: hai box **không giao nhau** thì IoU = 0
dù xa 1 pixel hay xa 500 pixel. Gradient = 0, model không biết phải dịch box về hướng nào.

CIoU (Complete IoU) thêm hai số hạng phạt:

$$
\text{CIoU} = \text{IoU} - \frac{\rho^2}{c^2} - \alpha v
$$

- `ρ²`: bình phương khoảng cách giữa **hai tâm**. `c²`: bình phương đường chéo của box nhỏ nhất bao cả hai. Tỉ số
  này kéo tâm lại gần nhau, kể cả khi hai box chưa giao.
- `v = (4/π²)(arctan(w₂/h₂) − arctan(w₁/h₁))²`: lệch tỉ lệ rộng/cao. `α = v / (v − IoU + 1)` cho số hạng này nặng
  hơn khi IoU đã cao (lúc đó mới đáng chỉnh dáng box).

**Ví dụ** với box gốc `[0, 0, 100, 100]` (đã tính bằng `bbox_ciou` của repo):

| Box thứ hai | IoU | CIoU | Đọc kết quả |
|---|---|---|---|
| `[0, 0, 100, 100]` | 1.000 | 1.000 | Trùng khớp |
| `[50, 0, 150, 100]` | 0.333 | 0.256 | Lệch tâm 50 px: phạt 2500 / 32500 = 0.077 |
| `[0, 0, 200, 50]` | 0.333 | 0.253 | Cùng IoU nhưng sai dáng: thêm phạt `αv` ≈ 0.018 |
| `[200, 0, 300, 100]` | **0.000** | **−0.400** | Không giao, IoU bó tay; CIoU vẫn chỉ hướng kéo về |

### Trong code

```python
weight = target_scores[fg].sum(-1)                 # ô dương "chắc chắn" hơn thì nặng hơn
ciou = bbox_ciou(pred_bboxes[fg], target_bboxes[fg])
loss_box = ((1 - ciou) * weight).sum() / ts_sum
```

Chỉ tính trên ô dương: ô nền không có box nào để khớp.

## 3. Loss phân bố: DFL (Distribution Focal Loss)

### Nhắc lại

Head không đoán thẳng khoảng cách `d` tới mỗi cạnh. Nó đoán một **phân bố** trên 16 bin 0, 1, …, 15, rồi `d` = kỳ
vọng của phân bố. Lý do: biên của vật thường mờ (bị che, bóng đổ), một phân bố diễn tả được "cạnh nằm đâu đó giữa 3
và 4" tốt hơn một con số duy nhất.

### Đích là hai bin kề nhau

Khoảng cách thật `t = 3.3` ô nằm giữa bin 3 và bin 4. DFL muốn phân bố dồn vào **đúng hai bin đó**, với trọng số theo
độ gần:

$$
\text{DFL} = -\left[ (4 - t) \log p_3 + (t - 3) \log p_4 \right] = -\left[ 0.7 \log p_3 + 0.3 \log p_4 \right]
$$

Loss nhỏ nhất khi `p₃ = 0.7, p₄ = 0.3`. Khi đó kỳ vọng = 3 × 0.7 + 4 × 0.3 = 3.3, đúng bằng đích.

| Phân bố dự đoán | DFL |
|---|---|
| `p₃ = 0.7, p₄ = 0.3`, còn lại 0 (tối ưu) | 0.611 |
| Đều 1/16 cho cả 16 bin | 2.773 |

DFL không về được 0: giá trị nhỏ nhất bằng entropy của cặp (0.7, 0.3). Vì vậy trong log train, `dfl` dừng quanh
0.75–0.8 dù box đã rất khớp. Đó là bình thường, không phải lỗi.

### Chi tiết

- Đích được kẹp vào `[0, 14.99]` để luôn có bin bên phải (`tl + 1 ≤ 15`). Khoảng cách xa hơn 15 ô không biểu diễn
  được ở mức đó; vật lớn thì để mức stride lớn lo.
- Tính trung bình 4 cạnh, nhân cùng `weight` như loss box, chia `ts_sum`.

## 4. Vì sao hệ số là 7.5, 0.5, 1.5

Ba loss có độ lớn tự nhiên rất khác nhau. Hệ số cân lại để không thành phần nào lấn át. Các con số này là
**hyperparameter** Ultralytics dò ra bằng thực nghiệm trên COCO, không phải suy ra từ lý thuyết. Với dataset khác, giữ
nguyên thường vẫn ổn.

Lúc train, `train.py` còn nhân tổng loss với batch size trước khi `backward()` (giống Ultralytics). Với AdamW, nhân
loss với hằng số gần như không đổi bước cập nhật, vì Adam chuẩn hoá theo độ lớn gradient.

## Tự kiểm tra

1. Nếu bỏ hẳn loss cls trên ô nền (chỉ tính ô dương), lúc dự đoán chuyện gì xảy ra? *(Mọi ô đều tự tin là có vật.)*
2. Tính DFL tối ưu cho `t = 5.0`. *(Toàn bộ xác suất ở bin 5, nên loss = 0.)*
3. Hai box cách nhau 1000 px và cách nhau 10 px, không giao nhau. `1 − IoU` và `1 − CIoU` của hai trường hợp khác nhau
   thế nào?
