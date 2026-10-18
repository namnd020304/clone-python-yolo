# 06. Đánh giá: mAP nghĩa là gì

**Mục tiêu:** hiểu con số mAP50 / mAP50-95 in ra trong log, và tự tính được bằng tay trên ví dụ nhỏ. Code:
[yolo/metrics.py](../../yolo/metrics.py). Chấm: `python check.py 15`.

## 1. Vì sao loss không đủ

Loss là thứ model *tối ưu*, không phải thứ ta *quan tâm*. Ta quan tâm: trên ảnh **chưa từng thấy** (tập val), model
tìm được bao nhiêu vật, và báo nhầm bao nhiêu. Loss train thấp có thể chỉ là học thuộc. Vì vậy cứ vài epoch, `train.py`
lại đo mAP trên tập val.

## 2. Đúng hay sai: TP và FP

Một dự đoán `[x1, y1, x2, y2, score, class]` là **TP** (true positive, đúng) khi:

1. cùng class với một vật thật, **và**
2. IoU với vật đó ≥ ngưỡng (ví dụ 0.5), **và**
3. vật đó chưa bị dự đoán khác ghép mất. Mỗi vật chỉ được tính đúng một lần; ưu tiên cặp IoU cao hơn.

Ngược lại là **FP** (false positive, báo nhầm). Vật thật không được dự đoán nào ghép thì là **FN** (bỏ sót).

`match_predictions` làm việc này cho cả 10 ngưỡng IoU 0.50, 0.55, …, 0.95 cùng lúc. Kết quả là ma trận `(K, 10)`
đúng/sai.

## 3. Precision và recall

$$
\text{precision} = \frac{TP}{TP + FP} \ \text{(báo ra thì đúng bao nhiêu)} \qquad
\text{recall} = \frac{TP}{\text{số vật thật}} \ \text{(tìm được bao nhiêu)}
$$

Hai số này kéo nhau: hạ ngưỡng điểm thì báo nhiều hơn, recall tăng nhưng precision thường giảm. Không có một ngưỡng
"đúng", nên ta xét **mọi ngưỡng**: xếp dự đoán theo điểm giảm dần, thêm từng cái một, mỗi lần ghi lại một điểm
(recall, precision). Nối lại được **đường precision-recall**.

## 4. AP: diện tích dưới đường precision-recall

**Ví dụ** một class, 2 vật thật, 3 dự đoán:

| Thứ tự | Điểm | Đúng? | TP cộng dồn | FP cộng dồn | Precision | Recall |
|---|---|---|---|---|---|---|
| 1 | 0.9 | TP | 1 | 0 | 1/1 = 1.00 | 1/2 = 0.5 |
| 2 | 0.8 | FP | 1 | 1 | 1/2 = 0.50 | 0.5 |
| 3 | 0.6 | TP | 2 | 1 | 2/3 = 0.67 | 2/2 = 1.0 |

Trước khi lấy diện tích, làm "phẳng" đường cong: precision tại recall r lấy bằng precision **lớn nhất** ở mọi recall
≥ r (`np.maximum.accumulate` chạy ngược). Ý nghĩa: nếu đi xa hơn mà precision còn cao hơn, ta luôn có thể chọn ngưỡng
đó.

```
precision
 1.00 ┤■■■■■■■■■■■■┐
      │            │
 0.67 ┤            └■■■■■■■■■■■■
      │
      └────────────┬────────────┬──► recall
                  0.5          1.0
```

Diện tích ≈ 0.5 × 1.00 + 0.5 × 0.67 = 0.833. Code lấy mẫu 101 điểm recall 0, 0.01, …, 1.0 (cách của COCO) rồi tính
hình thang, ra **0.828**. Hai số chênh nhau vì lấy mẫu rời rạc.

**AP tối đa là 0.995, không phải 1.** Đường cong rơi từ precision 1 xuống 0 ở recall = 1. Tính hình thang trên lưới
101 điểm thì ô cuối cùng chỉ được một nửa diện tích: 1 − 0.01 / 2 = 0.995. Ultralytics cũng cho đúng 0.995. Vì thế
thí nghiệm học thuộc 4 ảnh báo "mAP50-95 0.995" nghĩa là đã hoàn hảo.

## 5. Từ AP đến mAP50 và mAP50-95

- **AP** tính cho **một class**, ở **một ngưỡng IoU**.
- **mAP50**: trung bình AP của các class, ngưỡng IoU 0.5. Box chỉ cần trùng khoảng một nửa là tính đúng: chủ yếu đo
  "có tìm ra vật không".
- **mAP50-95**: trung bình thêm trên 10 ngưỡng 0.50 … 0.95. Ngưỡng 0.95 đòi box gần như khít tuyệt đối. Con số này đo
  cả "box có chính xác không", và là chỉ số chính của COCO. `best.pt` được chọn theo nó.

Ví dụ thật: `yolov8n.pt` gốc trên COCO val2017 đo bằng code repo được mAP50 0.454 nhưng mAP50-95 chỉ 0.347. Ở ngưỡng
IoU 0.5 model tìm ra gần một nửa số vật; đòi box khít hơn thì điểm tụt dần. (Thí nghiệm cũ trên biển báo: mAP50 0.716,
mAP50-95 0.581 sau 3 epoch.)

## 6. Chi tiết khi đo

- **Ngưỡng điểm rất thấp (0.001)**, khác hẳn 0.25 lúc dự đoán. Để đường precision-recall đi được tới recall cao nhất
  có thể: các dự đoán điểm thấp nằm cuối danh sách, chỉ kéo dài đường cong chứ không làm hỏng phần đầu.
- **NMS với IoU 0.7**, giữ tối đa 300 box mỗi ảnh (giống Ultralytics).
- **Gộp mọi ảnh rồi mới tính.** AP không phải trung bình của từng ảnh: mọi dự đoán của cả tập val được xếp chung theo
  điểm.

Kết quả đo `yolov8n.pt` trên coco8 val: repo này 0.758 / 0.600, Ultralytics 0.808 / 0.608 (mAP50 / mAP50-95). Phần
chênh có thể do Ultralytics cho một box giữ nhiều class khi đo, và resize ảnh bằng OpenCV. Với chỉ 17 vật, lệch một
dự đoán đã đổi mAP50 vài phần trăm.

## Tự kiểm tra

1. Đổi thứ tự trong ví dụ thành TP (0.9), TP (0.8), FP (0.6). AP lớn hơn hay nhỏ hơn 0.828? Vì sao?
2. Một model chỉ báo đúng 1 box rất chắc chắn và đúng, bỏ sót mọi vật khác. Precision, recall bằng bao nhiêu? AP cao
   hay thấp?
3. Vì sao mAP50-95 luôn ≤ mAP50?
