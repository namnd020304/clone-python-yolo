# Quy trình làm việc

## 1. Thế nào là "xong" một bước

Một bước trong plan chỉ được đánh dấu xong khi đủ cả 4 điều:

1. `python check.py` qua mọi bài (không chỉ bài của bước đó); bước mới có bài chấm riêng.
2. Đã chạy thật, không chỉ "code chạy được": ví dụ vẽ nhãn lên ảnh để xem box khớp vật, so số với Ultralytics, train
   ra mAP.
3. Tài liệu đã cập nhật:
   - `docs/ban-giao/trang-thai.md`: trạng thái bước, bảng kết quả đo;
   - `docs/ban-giao/nhat-ky.md`: một mục theo ngày, gồm đã làm gì, số đo, vấn đề gặp phải;
   - `docs/huong-dan/van-hanh.md`: nếu thêm hoặc đổi lệnh, tham số;
   - `docs/kien-truc/quyet-dinh.md`: nếu có quyết định mới hoặc đổi quyết định cũ;
   - `docs/train/`: nếu đổi code của giai đoạn train.
4. Đã commit.

Trung thực về trạng thái: code xong nhưng chưa chạy trên dữ liệu thật thì ghi "code xong, chưa đo", không ghi "xong".
Giả thuyết chưa kiểm chứng thì ghi là giả thuyết.

## 2. Quy tắc code (không vi phạm)

| Quy tắc | Lý do |
|---|---|
| `yolo/` không import Ultralytics | Mục tiêu là tự viết; Ultralytics chỉ để đối chiếu (QĐ-01) |
| Tên thuộc tính trong `yolo/` đổi thì sửa bảng `LAYERS` / `PARTS` trong `load_weights.py` | Nạp trọng số dựa vào tên (QĐ-04) |
| Docstring ghi rõ shape và đơn vị của tensor | Pixel và đơn vị ô dễ nhầm |
| Phần gán nhãn chạy trong `torch.no_grad()` | Assigner quyết định học cái gì, không phải thứ được học |
| Đo mAP và lưu checkpoint bằng bản EMA | Bản EMA ổn định hơn (QĐ-09) |
| Ví dụ số trong tài liệu phải được tính lại bằng code hoặc lấy từ log thật | Không bịa số |

## 3. Git

- Nhánh chính: `main`. Việc lớn thì tạo nhánh, ví dụ `feat/resume`.
- Commit message tiếng Anh, ví dụ `Add mosaic augmentation and EMA (step 16)`.
- Không commit: `venv/`, `datasets/`, `runs/`, `*.pt`, ảnh đầu ra. `.gitignore` đã chặn; xem `git status` trước khi commit.

## 4. Ngôn ngữ

- Tài liệu (`docs/`, README), docstring và comment: tiếng Việt (theo code hiện có).
- Tên biến, hàm, lớp: tiếng Anh.
- Commit message: tiếng Anh.

## 5. Làm việc với AI hỗ trợ code (Claude Code)

- Giao việc theo bước của plan, kèm bài chấm. Người phải đọc diff và chạy `check.py`; không nhận code chưa chạy.
- Muốn tự học thì yêu cầu AI tạo khung `raise NotImplementedError` + docstring + bài chấm, mình tự viết phần thân.
- Lặp trên dữ liệu nhỏ (coco8, 4 ảnh) trước; chỉ chạy dataset thật khi đã ổn.
- Việc tốn tiền (máy ảo GCP) phải xác nhận tài khoản, project, cấu hình trước khi tạo; xong phải xoá.
- Cuối buổi, yêu cầu AI cập nhật `trang-thai.md` và `nhat-ky.md`.
