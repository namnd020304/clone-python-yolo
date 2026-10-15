# Tài liệu dự án — đọc file này trước

Repo này viết lại YOLOv8n từ số 0 bằng PyTorch để học: dựng model, train, đánh giá, chạy thật. Tài liệu được viết để
người mới (hoặc chính mình sau vài tháng) chỉ đọc repo là làm tiếp được, không cần hỏi lại. Nếu thấy điều gì phải hỏi
mới biết, hãy bổ sung vào tài liệu tương ứng sau khi đã hỏi được.

## Thứ tự đọc cho người mới tiếp nhận

| Bước | Đọc | Mục đích | Thời gian |
|---|---|---|---|
| 1 | [../README.md](../README.md) | Dự án là gì, 9 bước dựng model, con số cần nhớ | 5 phút |
| 2 | [ke-hoach-hoc.md](ke-hoach-hoc.md) | Plan học: 3 giai đoạn, từng bước học gì, kiểm chứng bằng gì | 15 phút |
| 3 | [ban-giao/trang-thai.md](ban-giao/trang-thai.md) | Đang ở đâu: bước nào xong, đang làm, chưa làm | 10 phút |
| 4 | [ban-giao/van-de-mo.md](ban-giao/van-de-mo.md) | Việc còn bỏ ngỏ, điều chưa giải thích được | 5 phút |
| 5 | [kien-truc/tong-quan-kien-truc.md](kien-truc/tong-quan-kien-truc.md) | Module nào làm gì, luồng suy luận và luồng train | 20 phút |
| 6 | [kien-truc/quyet-dinh.md](kien-truc/quyet-dinh.md) | Vì sao code như hiện tại, khi nào nên xem lại | 15 phút |
| 7 | [huong-dan/cai-dat-moi-truong.md](huong-dan/cai-dat-moi-truong.md) | Cài máy, chạy `check.py` tới khi qua 16/16 | 30 phút |
| 8 | [huong-dan/quy-trinh-lam-viec.md](huong-dan/quy-trinh-lam-viec.md) | Thế nào là xong, quy ước code, git, cập nhật tài liệu | 10 phút |
| 9 | [train/README.md](train/README.md) | Lý thuyết giai đoạn train (bước 10–16), đọc khi cần đào sâu | 2–3 giờ |
| 10 | [ban-giao/nhat-ky.md](ban-giao/nhat-ky.md) | Lịch sử: đã thử gì, số đo, vì sao đổi | Khi cần |

## Bản đồ tài liệu

| File | Nội dung | Cập nhật khi nào |
|---|---|---|
| [ke-hoach-hoc.md](ke-hoach-hoc.md) | Plan học: mục tiêu, 3 giai đoạn, bước 1–16 và phần còn lại, học song song C++ | Khi đổi lộ trình. Là căn cứ gốc |
| [huong-dan/cai-dat-moi-truong.md](huong-dan/cai-dat-moi-truong.md) | Từ máy trắng đến chạy được `check.py`, lỗi thường gặp | Khi thêm thư viện, công cụ |
| [huong-dan/van-hanh.md](huong-dan/van-hanh.md) | Các lệnh: chấm, nạp trọng số, lấy dữ liệu, train, dự đoán, camera, train trên GCP | Khi thêm/đổi lệnh hoặc tham số |
| [huong-dan/quy-trinh-lam-viec.md](huong-dan/quy-trinh-lam-viec.md) | Thế nào là "xong", quy tắc code, git, làm với AI | Khi đổi quy ước |
| [kien-truc/tong-quan-kien-truc.md](kien-truc/tong-quan-kien-truc.md) | Module, shape tensor qua từng khâu, nơi lưu file | Khi thêm module hoặc đổi luồng |
| [kien-truc/quyet-dinh.md](kien-truc/quyet-dinh.md) | Nhật ký quyết định (QĐ-01…): bối cảnh, lựa chọn, khi nào xem lại | Mỗi khi chốt/đổi một quyết định |
| [train/](train/README.md) | Lý thuyết từng bước train (01–07), có ví dụ số và câu tự kiểm tra | Khi code train đổi |
| [ban-giao/trang-thai.md](ban-giao/trang-thai.md) | Bước xong / đang làm / chưa làm, bảng kết quả đo | Cuối mỗi buổi làm |
| [ban-giao/nhat-ky.md](ban-giao/nhat-ky.md) | Nhật ký theo ngày | Cuối mỗi buổi làm |
| [ban-giao/van-de-mo.md](ban-giao/van-de-mo.md) | Câu hỏi chưa có lời đáp, việc chờ quyết | Khi có câu hỏi mới hoặc đã chốt |
| [ban-giao/checklist-ban-giao.md](ban-giao/checklist-ban-giao.md) | Danh sách phải làm trước khi giao repo cho người khác | Trước khi bàn giao |
| [bao-cao-yolov8n.docx](bao-cao-yolov8n.docx) | Báo cáo toàn bộ quy trình (12/10/2026), viết cho người đọc không cần xem code | Khi cần báo cáo lại |

Khi tài liệu mâu thuẫn nhau: quyết định mới nhất trong `kien-truc/quyet-dinh.md` và `ban-giao/nhat-ky.md` thắng,
sau đó đến `ke-hoach-hoc.md`, sau đó đến báo cáo `.docx` (bản chụp ngày 12/10). Thấy mâu thuẫn thì sửa ngay chỗ cũ.
Con số đo luôn kèm ngày; số mới thay số cũ chứ không xoá số cũ khỏi nhật ký.
