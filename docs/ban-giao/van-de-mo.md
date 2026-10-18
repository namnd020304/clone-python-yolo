# Vấn đề mở và việc chờ chốt

Mỗi dòng có người quyết (hoặc người làm) và việc cần làm để đóng. Đã chốt thì chuyển xuống mục "Đã chốt", ghi kết
quả và ngày; nếu là quyết định thiết kế thì thêm vào [../kien-truc/quyet-dinh.md](../kien-truc/quyet-dinh.md).

## Đang mở

| # | Vấn đề | Ai quyết / làm | Cần gì để đóng | Ảnh hưởng nếu để đó |
|---|---|---|---|---|
| 2 | Train từ số 0 trên 4 ảnh: mAP = 0 tới epoch 100 dù loss đã thấp, rồi tăng vọt ở epoch 150–200. EMA không đổi hiện tượng này | Chủ repo | Đo IoU trung bình của box dự đoán và điểm class cao nhất theo từng epoch | Không chặn gì (model vẫn học được); là câu hỏi để hiểu sâu |
| 4 | `webcam.py` chưa có người xem cửa sổ chạy thật | Chủ repo | Chạy `python webcam.py`, ghi FPS và nhận xét vào nhật ký | Có thể còn lỗi hiển thị chưa thấy |
| 5 | Báo cáo `.docx` chưa xem giao diện trong Word (máy không có Word / LibreOffice để render) | Chủ repo | Mở file, sửa bảng tràn cột hoặc ngắt trang xấu | Báo cáo trình bày xấu |
| 8 | Sau 30 epoch từ số 0 trên GPU: mAP50-95 **0.261** (mốc 0.347, ~75%), chưa hội tụ. Có train dài hơn không, và so với job GCP | Chủ repo | Quyết số epoch / lịch lr cho lần train tiếp; lấy kết quả job GCP để so | Model hiện dưới bản gốc rõ rệt |
| 11 | Epoch chậm gần gấp đôi từ epoch 16 (~930 s → ~1700 s), không rõ vì sao | Chủ repo | Theo dõi nhiệt độ/xung GPU-CPU, tải CPU, chế độ nguồn khi train lại | Lần train sau tốn ~11,6 giờ thay vì ~8 |
| 9 | `yolov8n.pt` đo bằng code repo trên val2017 được 0.347, Ultralytics công bố 0.373 | Chủ repo | So cách đo: ảnh vuông 640 hay giữ tỉ lệ, ngưỡng conf, số box tối đa | mAP của model tự train có thể cũng bị đo thấp hơn thực tế |
| 7 | Nhận diện "đúng là mình" (face recognition) khác với phát hiện "có người". Repo này chỉ làm phát hiện | Chủ repo | Quyết có làm không; nếu làm thì ở repo `face-regconition` cạnh đây | Kỳ vọng sai về `webcam.py` |

## Đã chốt

| # | Vấn đề | Kết quả | Ngày |
|---|---|---|---|
| — | Train trên GPU hay CPU | CPU trên máy ảo GCP vì tài khoản free trial không có hạn mức GPU (QĐ-08) | 11/10/2026 |
| — | lr mặc định | `0.002 × 5 / (4 + nc)` như Ultralytics (QĐ-06) | 11/10/2026 |
| — | Dataset riêng để ở đâu | Trong `datasets/` của repo (đã gitignore); dataset rất lớn thì để ngoài và truyền đường dẫn tuyệt đối | 11/10/2026 |
| 1, 3 | EMA + mosaic và vật nhỏ trên biển báo | Bỏ: không dùng biển báo nữa, chỉ train COCO từ số 0 (QĐ-10). Đánh giá EMA + mosaic bằng kết quả COCO | 16/10/2026 |
| 6 | Resume | Đã có `--resume` | 15/10/2026 |
| 10 | `pyproject.toml`, `uv.lock` nằm ngoài repo | Đã chuyển vào repo; `git clone` lấy đủ cấu hình môi trường | 16/10/2026 |
