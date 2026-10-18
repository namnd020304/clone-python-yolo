# 05. Vòng lặp train: biến loss thành trọng số tốt hơn

**Mục tiêu:** hiểu từng dòng trong vòng lặp train, cách chọn learning rate, và khác biệt giữa train từ đầu với
fine-tune. Code: [train.py](../../train.py). Chấm: `python check.py 14`.

## 1. Một bước train

```python
for imgs, gt_labels, gt_bboxes, mask in train_loader:      # 1 batch
    lr = lr_at(it, total, warmup, args.lr)                    # 4. lịch learning rate
    loss = criterion(model(imgs), gt_labels, gt_bboxes, mask) # forward: dự đoán → gán nhãn → loss
    optimizer.zero_grad()                                     # xoá gradient của bước trước
    (loss.sum() * len(imgs)).backward()                       # backward: tính gradient mọi tham số
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)        # 3. chặn gradient quá lớn
    optimizer.step()                                          # 2. cập nhật trọng số
```

- **Forward:** chạy model, ghi lại mọi phép tính thành một đồ thị.
- **Backward:** đi ngược đồ thị, tính `∂loss/∂w` cho từng trọng số `w` (quy tắc chuỗi, autograd làm hộ).
- **Step:** dịch mỗi `w` một chút ngược hướng gradient, để loss giảm.
- `zero_grad()` bắt buộc phải có: PyTorch **cộng dồn** gradient qua các lần `backward()`.

Thuật ngữ: một **iteration** là một batch. Một **epoch** là đi hết tập train một lượt. COCO train2017 có 118.287
ảnh, batch 16, nên 7.393 iteration mỗi epoch (thanh tiến trình ghi `.../7393`).

## 2. Optimizer: AdamW

SGD thuần dịch mọi trọng số cùng một tỉ lệ: `w ← w − lr · g`. **Adam** giữ trung bình trượt của gradient (`m`, như
quán tính) và của bình phương gradient (`v`), rồi dịch theo `m / √v`. Kết quả là mỗi trọng số có bước đi riêng, xấp
xỉ `lr`, bất kể gradient của nó to hay nhỏ. Adam hội tụ nhanh và ít phải chỉnh tay.

**W** trong AdamW là **weight decay** tách riêng: mỗi bước kéo trọng số về 0 một chút (`5e-4`), để model không dựa vào
vài trọng số cực lớn. Đây là một dạng chống overfit.

Weight decay chỉ áp cho trọng số **conv**. BatchNorm (γ, β) và bias thì không, vì kéo chúng về 0 không có ý nghĩa
gì, chỉ làm hại. Hàm `build_optimizer` chia tham số thành hai nhóm như vậy.

## 3. Cắt gradient (gradient clipping)

Thỉnh thoảng một batch lạ cho gradient cực lớn và một bước cập nhật phá hỏng model. `clip_grad_norm_(…, 10)`: nếu độ
dài (norm) của toàn bộ vector gradient lớn hơn 10 thì co nó về đúng 10, giữ nguyên hướng.

## 4. Learning rate: con số quan trọng nhất

`lr` quyết định mỗi bước đi xa bao nhiêu. Quá nhỏ thì học chậm; quá lớn thì nhảy qua chỗ tốt, hoặc phá hỏng hẳn.

### Bài học thật từ repo này

Lần đầu fine-tune `yolov8n.pt` trên coco8 với `lr = 0.002`, mAP50-95 rơi từ **0.600 xuống 0.000** chỉ sau vài bước,
dù loss train vẫn giảm. Với Adam, mỗi trọng số dịch khoảng `lr` mỗi bước. Trọng số đã học kỹ không chịu nổi cú dịch
0.002 trên mọi tham số cùng lúc.

Sửa: lấy công thức "tự chọn" của Ultralytics:

$$
\text{lr} = 0.002 \cdot \frac{5}{4 + n_c}
$$

| nc | lr |
|---|---|
| 80 (COCO) | 0.000119 |
| 4 (biển báo) | 0.00125 |

Có lr này, fine-tune giữ được rồi còn tăng mAP (0.600 → 0.636).

### Train từ số 0 cần lr lớn hơn

Công thức trên dành cho fine-tune và job ngắn: trọng số đã tốt, chỉ cần chỉnh nhẹ. Train từ số 0 thì ngược lại, trọng
số ngẫu nhiên phải đi rất xa. Với 80 class công thức cho 0.000119, quá nhỏ. Job COCO từ số 0 của repo dùng
`--lr 0.001` (Ultralytics với job dài dùng SGD lr 0.01; Adam đi bước lớn hơn SGD với cùng lr nên cần lr nhỏ hơn).
Con số 0.001 chưa được so với giá trị khác; khi có kết quả 30 epoch sẽ xem lại.

### Warmup rồi giảm dần theo cosine

```
lr
 ▲         ______
 │       /        ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\___
 │     /                                         ‾‾‾‾\____
 │   /                                                     ‾‾‾‾‾‾‾____
 └─/───────────────────────────────────────────────────────────────────► iteration
   warmup                  cosine giảm về lr × 0.01
```

- **Warmup** (`min(max(3 epoch, 10 iteration), 1/5 tổng)`): lr tăng tuyến tính từ gần 0. Lúc đầu, thống kê của Adam
  (`m`, `v`) chưa ổn định và BatchNorm chưa quen dữ liệu mới; bước nhỏ giúp không phá model.
- **Cosine:** lr giảm mượt về 1% lúc cuối. Bước lớn giúp đi nhanh lúc đầu, bước nhỏ giúp tinh chỉnh lúc cuối.

## 5. BatchNorm ở hai chế độ

Lúc `train()`, BatchNorm chuẩn hoá bằng trung bình, phương sai **của batch hiện tại**, đồng thời cập nhật trung bình
trượt (`momentum = 0.03`, tức mỗi batch đóng góp 3%). Lúc `eval()`, nó dùng trung bình trượt đó.

Hệ quả thấy trong thí nghiệm train từ số 0 (chưa có EMA): ở epoch 50 và 100, loss train đã thấp nhưng mAP đo ở chế độ
eval vẫn **0.000**, rồi tới epoch 200 mới vọt lên 0.995. Một giải thích có thể (chưa kiểm chứng): trọng số đổi nhanh
hơn trung bình trượt theo kịp, nên ở chế độ eval model chạy với thống kê "cũ".

**Đã thử lại có EMA, hiện tượng vẫn y như cũ:** mAP = 0.000 ở epoch 50 và 100, 0.096 ở epoch 150, 0.990 ở epoch 200.
Vậy giả thuyết BatchNorm ở trên không được ủng hộ, và EMA không phải thứ sửa được chuyện này. Nguyên nhân thật chưa
được xác định. Một hướng khả dĩ: khi train từ đầu, box dự đoán mất nhiều epoch mới đạt IoU ≥ 0.5 với vật thật, nên
dù loss đã giảm, chưa dự đoán nào được tính là đúng. Cần đo IoU trung bình của ô dương theo epoch mới kết luận được.

## 5b. EMA: đo và lưu bản "trung bình" của model

Code: [yolo/ema.py](../../yolo/ema.py). Chấm: `python check.py 16`.

Sau mỗi bước, trọng số dao động quanh điểm tốt, vì mỗi batch kéo một hướng hơi khác. EMA giữ một **bản sao** model
có trọng số là trung bình trượt của các bước gần đây:

$$
w_{\text{ema}} \leftarrow d \cdot w_{\text{ema}} + (1 - d) \cdot w, \qquad d = 0.9999 \cdot \left(1 - e^{-\text{bước} / 2000}\right)
$$

- `d` tăng dần từ 0 lên 0.9999. Lúc đầu model đổi nhanh, `d` nhỏ nên EMA bám sát; về sau EMA "nhớ" khoảng 10 nghìn
  bước gần nhất. Ở bước 2000, `d ≈ 0.63`; ở bước 10000, `d ≈ 0.993`.
- Running mean, running var của BatchNorm cũng được lấy trung bình y như vậy.
- Gradient và optimizer **chỉ** tác động lên model chính. EMA chỉ "đi theo".
- `train.py` đo mAP và lưu `best.pt` / `last.pt` bằng bản EMA. Tắt bằng `--no-ema`.

Lấy trung bình nhiều điểm quanh đáy thường cho một điểm gần đáy hơn từng điểm riêng lẻ. Vì vậy mAP đo bằng EMA thường
cao hơn và dao động ít hơn giữa các lần đo.

## 6. Train từ đầu và fine-tune

Repo chọn **train từ đầu** (QĐ-10): mục tiêu là tự build model, không phải chỉnh model của người khác.

| | Train từ đầu | Fine-tune từ `yolov8n.pt` |
|---|---|---|
| Khởi đầu | Trọng số ngẫu nhiên + bias đặc biệt (file 02) | Trọng số đã học trên 118 nghìn ảnh COCO |
| Dữ liệu cần | Rất nhiều (hàng chục nghìn ảnh) | Vài trăm ảnh là dùng được |
| Thời gian | Ultralytics: 500 epoch COCO trên nhiều GPU | Vài chục epoch |
| Repo này | 4 ảnh: học thuộc được, không tổng quát. COCO 118 nghìn ảnh: đang làm | Biển báo 3 epoch: mAP50 0.716 (hướng cũ, đã bỏ) |

Backbone đã học trên COCO biết "nhìn" cạnh, góc, kết cấu, hình dạng. Những thứ đó dùng được cho gần như mọi bài toán
ảnh. Fine-tune chỉ phải dạy phần cuối, nên nhanh, nhưng phần quý nhất của model lại do người khác train. Train từ
đầu trên COCO tốn hơn nhiều (Ultralytics: 500 epoch) nhưng mọi trọng số đều do code của repo học ra.

### Phép thử "học thuộc" (overfit test)

Trước khi train lớn, hãy thử cho model học thuộc vài ảnh: train và đo trên **cùng** tập đó, tắt augmentation. Nếu
đến đó mà loss không giảm, hoặc mAP không gần 1, thì code có lỗi (sai nhãn, sai loss, sai assigner). Đó chính là thí
nghiệm 300 epoch trên 4 ảnh coco8: mAP50-95 đạt 0.995. `check.py 14` là bản thu nhỏ: 15 bước trên 1 ảnh, loss phải
giảm hơn một nửa.

```powershell
python train.py --epochs 300 --lr 0.002 --val-split train --no-augment --eval-every 50
```

## 7. Đọc log train

```
epoch    3/3  box 0.752  cls 0.875  dfl 1.000  lr 0.00001  185.5s  |  mAP50 0.716  mAP50-95 0.581
```

- `box`, `cls`, `dfl`: loss trung bình của epoch, đã nhân hệ số. Phải giảm dần, có dao động.
- `dfl` dừng quanh 0.75–0.8 là bình thường (xem file 04).
- Loss train giảm mà mAP val giảm theo là dấu hiệu **overfit**. `best.pt` lưu lại epoch có mAP50-95 val cao nhất,
  nên vẫn lấy được model tốt nhất.

## Tự kiểm tra

1. Quên `optimizer.zero_grad()` thì chuyện gì xảy ra sau 10 bước?
2. Vì sao train từ số 0 cần lr lớn hơn fine-tune? Công thức cho fine-tune 4 class lr lớn gấp hơn 10 lần so với 80 class. Công thức này là kinh nghiệm của
   Ultralytics, không phải định lý. Thử nghĩ một lý do hợp lý. *(Gợi ý: với 4 class, nhánh class được học lại từ đầu,
   còn với 80 class mọi trọng số đều đã tốt sẵn.)*
3. Log ghi loss train giảm đều, mAP val tăng tới epoch 10 rồi giảm. Nên dùng checkpoint nào?
