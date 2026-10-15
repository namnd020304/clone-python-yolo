# Vấn đề mở và việc chờ chốt

Mỗi dòng có người quyết (hoặc người làm) và việc cần làm để đóng. Đã chốt thì chuyển xuống mục "Đã chốt", ghi kết
quả và ngày; nếu là quyết định thiết kế thì thêm vào [../kien-truc/quyet-dinh.md](../kien-truc/quyet-dinh.md).

## Đang mở

| # | Vấn đề | Ai quyết / làm | Cần gì để đóng | Ảnh hưởng nếu để đó |
|---|---|---|---|---|
| 1 | EMA + mosaic có tăng mAP trên dữ liệu thật không. Số 0.705 là của bản chưa có hai tính năng này | Chủ repo | Train lại `bien-bao` (~3 giờ CPU GCP, ~1,3 USD credit) rồi so; xem riêng biển nhỏ | Không biết bước 16 có đáng giữ mặc định bật không |
| 2 | Train từ số 0 trên 4 ảnh: mAP = 0 tới epoch 100 dù loss đã thấp, rồi tăng vọt ở epoch 150–200. EMA không đổi hiện tượng này | Chủ repo | Đo IoU trung bình của box dự đoán và điểm class cao nhất theo từng epoch | Không chặn gì (model vẫn học được); là câu hỏi để hiểu sâu |
| 3 | Vật nhỏ (dưới ~10 px) bị bỏ sót | Chủ repo | Đo lại sau #1; nếu vẫn kém thử `--size` lớn hơn hoặc thêm co giãn | Model biển báo kém với biển ở xa |
| 4 | `webcam.py` chưa có người xem cửa sổ chạy thật | Chủ repo | Chạy `python webcam.py`, ghi FPS và nhận xét vào nhật ký | Có thể còn lỗi hiển thị chưa thấy |
| 5 | Báo cáo `.docx` chưa xem giao diện trong Word (máy không có Word / LibreOffice để render) | Chủ repo | Mở file, sửa bảng tràn cột hoặc ngắt trang xấu | Báo cáo trình bày xấu |
| 6 | Chưa có resume (train tiếp từ `last.pt`) | Chủ repo | Lưu thêm optimizer, epoch, EMA vào checkpoint; thêm `--resume` | Máy ảo spot bị thu hồi là phải train lại từ đầu |
| 7 | Nhận diện "đúng là mình" (face recognition) khác với phát hiện "có người". Repo này chỉ làm phát hiện | Chủ repo | Quyết có làm không; nếu làm thì ở repo `face-regconition` cạnh đây | Kỳ vọng sai về `webcam.py` |

## Đã chốt

| # | Vấn đề | Kết quả | Ngày |
|---|---|---|---|
| — | Train trên GPU hay CPU | CPU trên máy ảo GCP vì tài khoản free trial không có hạn mức GPU (QĐ-08) | 11/10/2026 |
| — | lr mặc định | `0.002 × 5 / (4 + nc)` như Ultralytics (QĐ-06) | 11/10/2026 |
| — | Dataset riêng để ở đâu | Trong `datasets/` của repo (đã gitignore); dataset rất lớn thì để ngoài và truyền đường dẫn tuyệt đối | 11/10/2026 |
