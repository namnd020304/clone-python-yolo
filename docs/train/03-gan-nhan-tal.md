# 03. Gán nhãn: ô nào học vật nào (TAL)

**Mục tiêu:** hiểu bài toán gán nhãn và cách Task-Aligned Assigner giải nó. Đây là phần khó nhất của cả quá trình
train. Code: [yolo/assigner.py](../../yolo/assigner.py). Chấm: `python check.py 12` (so với Ultralytics).

## 1. Bài toán

Model đưa ra 8400 dự đoán. Ảnh có, ví dụ, 3 vật thật. Muốn tính loss, ta phải trả lời: **dự đoán nào nên khớp với vật
nào?**

- Ô nào được giao một vật thì gọi là ô **dương** (foreground). Nó phải học đúng box và class của vật đó.
- Mọi ô còn lại là **nền** (background). Chúng chỉ phải học "điểm mọi class = 0".

Nếu giao sai, ví dụ bắt một ô ở góc ảnh học một vật ở giữa ảnh, model sẽ học một việc vô lý. Gán nhãn tốt là nửa thành
công của detector.

Các cách đã có trong lịch sử:

| Cách | Ý tưởng | Nhược điểm |
|---|---|---|
| YOLOv3–v5: theo anchor box | Ô chứa tâm vật + anchor có hình gần giống | Phải chọn tay kích thước anchor |
| FCOS, ATSS: theo hình học | Ô nằm trong vật, gần tâm vật | Không quan tâm model đang đoán tốt hay dở |
| **TAL (YOLOv8)** | Ô nằm trong vật **và** đang đoán tốt cả class lẫn vị trí | Phụ thuộc dự đoán hiện tại, nên phải chạy lại mỗi bước |

## 2. Điểm "khớp nhiệm vụ" (align metric)

Với mỗi cặp (vật thật g, ô a):

$$
\text{align}(g, a) = \underbrace{s(a, c_g)^{\alpha}}_{\text{đoán đúng class}} \cdot \underbrace{\text{CIoU}(g, \hat{b}_a)^{\beta}}_{\text{đoán đúng vị trí}}, \qquad \alpha = 0.5,\ \beta = 6
$$

- `s(a, c_g)`: điểm (đã qua sigmoid) mà ô a đang đoán cho **đúng class của g**.
- `b̂_a`: box mà ô a đang dự đoán. CIoU đo nó khớp với g tới đâu (file 04).

"Task-aligned" nghĩa là: ô tốt phải giỏi **cả hai nhiệm vụ** cùng lúc. Một ô đoán class rất chắc nhưng box lệch, hoặc
box khớp nhưng không biết là class gì, đều bị điểm thấp.

**β = 6 rất lớn**, nên vị trí áp đảo:

| Ô | điểm class | CIoU | align |
|---|---|---|---|
| A | 0.5 | 0.8 | 0.5^0.5 × 0.8^6 = 0.707 × 0.262 = **0.185** |
| B | 0.9 | 0.6 | 0.9^0.5 × 0.6^6 = 0.949 × 0.047 = **0.044** |

Ô A thắng dù đoán class kém hơn, vì box của nó khớp hơn nhiều.

## 3. Thuật toán, từng bước

Tất cả tính bằng pixel ảnh 640, trong `torch.no_grad()`.

**Bước 1. Ứng viên: tâm ô phải nằm trong vật.** `in_gts (B, M, 8400)`. Vật rất nhỏ (cạnh < 16 px) có thể không chứa
tâm ô nào, nên được nới ra thành 16 px quanh tâm của nó trước khi xét. Đây là chi tiết của Ultralytics 8.4; YOLOv8 bản
đầu không có.

**Bước 2. Tính align** cho mọi cặp, nhân `in_gts` để cặp ngoài vật bằng 0.

**Bước 3. Top-k.** Mỗi vật lấy 10 ô có align cao nhất (`topk = 10`). Kết hợp với bước 1: `mask_pos = top-10 VÀ nằm
trong vật`.

**Bước 4. Giải tranh chấp.** Hai vật chồng lên nhau có thể cùng chọn một ô. Một ô chỉ học được một vật, nên ô đó thuộc
về vật có **CIoU cao nhất** với box ô đang đoán.

```python
multi = mask_pos.sum(1, keepdim=True) > 1          # ô bị >1 vật chọn
best_gt = one_hot(ious.max(1).indices)             # vật có CIoU cao nhất ở ô đó
mask_pos = torch.where(multi, best_gt, mask_pos)
fg_mask, target_gt_idx = mask_pos.max(1)           # mỗi ô: có phải dương không, thuộc vật nào
```

**Bước 5. Lấy box và class** của vật được giao cho từng ô.

**Bước 6. Nhãn class "mềm".** Đích của BCE ở ô dương **không phải 1**, mà là:

$$
\text{target}(a) = \frac{\text{align}(g, a)}{\max_{a'} \text{align}(g, a')} \cdot \max_{a'} \text{CIoU}(g, a')
$$

Nghĩa là: ô tốt nhất của vật g có đích bằng CIoU tốt nhất mà model đang đạt cho vật đó; các ô kém hơn có đích nhỏ hơn
theo tỉ lệ. Ô nền có đích 0.

Vì sao không đặt đích = 1? Vì khi đó điểm class không nói gì về chất lượng box. Với nhãn mềm, **điểm class học cách
phản ánh luôn độ chính xác của box**: box càng khớp thì điểm càng cao. Lúc dự đoán, ta xếp hạng và lọc theo điểm này,
nên box tốt tự nổi lên trên. NMS cũng hưởng lợi: box có điểm cao nhất thường là box khớp nhất.

## 4. Vì sao không có gradient đi qua assigner

Assigner chỉ quyết định **học cái gì** (đích). Nếu gradient đi qua, model có thể "ăn gian" bằng cách đổi dự đoán để
đích dễ đạt hơn, thay vì học đúng. Nên mọi đầu vào được `.detach()` và hàm chạy trong `@torch.no_grad()`.

## 5. Shape cần nhớ

| Tensor | Shape | Ý nghĩa |
|---|---|---|
| `pd_scores` | (B, 8400, nc) | Điểm class đã sigmoid |
| `pd_bboxes` | (B, 8400, 4) | Box dự đoán, pixel xyxy |
| `in_gts`, `align`, `ious`, `mask_pos` | (B, M, 8400) | Mỗi cặp (vật, ô) |
| `fg_mask` | (B, 8400) | Ô có phải dương không |
| `target_bboxes` | (B, 8400, 4) | Box đích của từng ô |
| `target_scores` | (B, 8400, nc) | Nhãn class mềm |

Mẹo đọc code: hầu hết phép tính là trên tensor `(B, M, 8400)`. Mỗi lần thấy `.sum(1)`, `.max(1)` là đang gộp theo
**vật** (trục M) để ra một giá trị cho mỗi **ô**; `.amax(-1)`, `.topk(dim=-1)` là gộp theo **ô** để ra một giá trị cho
mỗi vật.

## Tự kiểm tra

1. Đổi `beta` từ 6 xuống 1 thì ô được chọn sẽ thay đổi thế nào?
2. Một vật lớn chứa 500 tâm ô. Bao nhiêu ô trong số đó thành ô dương? *(Tối đa 10, có thể ít hơn nếu tranh chấp.)*
3. Ở đầu quá trình train từ số 0, mọi điểm class ≈ 0 và box gần như ngẫu nhiên. TAL có còn chọn được ô hợp lý không?
   Gợi ý: điều kiện "nằm trong vật" vẫn giữ, và ô gần tâm vật thường có CIoU cao hơn.
