# Checklist bàn giao

Dùng khi giao repo cho người khác, hoặc khi chính mình quay lại sau thời gian dài. Điền bảng kiểm kê ngay khi tạo tài
nguyên, không để đến lúc bàn giao.

## 1. Kiểm kê tài nguyên

| Loại | Tên / ID | Owner | Ghi chú |
|---|---|---|---|
| Repo git (remote) | `github.com/namnd020304/clone-python-yolo` | namnd020304 | Nhánh `main` |
| Trọng số gốc | `yolov8n.pt` (Ultralytics release v8.3.0) | Ultralytics | Không commit (`*.pt` trong `.gitignore`); tải lại theo `huong-dan/cai-dat-moi-truong.md` |
| Trọng số COCO đổi định dạng | `yolov8n_mine.pt` | Chủ repo | Tạo từ `yolov8n.pt` bằng `train.load_pretrained`; `webcam.py` dùng mặc định |
| Model biển báo | `runs/bien-bao/best.pt` (+ `train.log`) | Chủ repo | Chỉ có trên máy; không commit. Cần lưu thêm ở chỗ khác nếu muốn giữ |
| Dataset | `datasets/coco8`, `datasets/bien-bao` | Ultralytics / Kaggle | Tải lại được bằng lệnh, không cần lưu |
| Project GCP | Project dùng thử (free trial) của chủ repo | Chủ repo | VM `yolo-train` **đã xoá** 12/10; không còn tài nguyên tính tiền |
| Tài liệu Docs giai đoạn 1 | "Tự dựng lại YOLOv8n từ số 0" trên claude.ai | Chủ repo | Bản gốc trên claude.ai; nội dung chính đã có trong `README.md` và `ke-hoach-hoc.md` |
| Báo cáo | `docs/bao-cao-yolov8n.docx` (+ bản Docs trên claude.ai) | Chủ repo | |

Không có secret nào trong repo. Nếu dùng dataset Kaggle cần đăng nhập thì `kaggle.json` nằm ở
`C:/Users/<bạn>/.kaggle/`, không đặt trong repo.

## 2. Trước khi bàn giao

- [ ] Commit hết; `git status` sạch (trừ file đã gitignore)
- [ ] `python check.py` qua 16/16 trên máy sạch
- [ ] `ban-giao/trang-thai.md` và `nhat-ky.md` đúng thực tế
- [ ] `van-de-mo.md`: mỗi vấn đề có người làm hiện tại
- [ ] `kien-truc/quyet-dinh.md` có mọi thay đổi so với plan
- [ ] `huong-dan/van-hanh.md` đủ lệnh và tham số
- [ ] Bảng kết quả đo có số thật, kèm ngày
- [ ] `runs/bien-bao/best.pt` được gửi cho người tiếp nhận (không nằm trong git)

## 3. Người tiếp nhận tự kiểm chứng (chỉ đọc tài liệu, không hỏi)

Bước nào phải hỏi thì sửa tài liệu chỗ đó.

- [ ] Cài môi trường, `python check.py` qua 16/16
- [ ] `python load_weights.py` ra `result.jpg` có 4 người + 1 xe buýt, lệch ≤ 1e-4
- [ ] Tải `bien-bao` bằng `kaggle_data.py`, fine-tune 3 epoch, mAP50 lên quanh 0.7
- [ ] `predict.py` trên một ảnh val, box đúng biển
- [ ] `webcam.py` mở camera, có box quanh người

## 4. Buổi chuyển giao

1. Tổng quan và đọc tài liệu theo `docs/README.md` (1 giờ).
2. Người tiếp nhận tự cài và chạy, người cũ chỉ sửa tài liệu (1 giờ).
3. Đi qua code: blocks → model → head (DFL) → TAL → loss → train (1–2 giờ). TAL và loss là phần khó nhất.
4. Người tiếp nhận làm một việc nhỏ trọn vẹn đến commit, ví dụ thêm `--resume` (nửa ngày).
