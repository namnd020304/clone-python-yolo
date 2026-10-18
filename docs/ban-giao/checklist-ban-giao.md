# Checklist bàn giao

Dùng khi giao repo cho người khác, hoặc khi chính mình quay lại sau thời gian dài. Điền bảng kiểm kê ngay khi tạo tài
nguyên, không để đến lúc bàn giao.

## 1. Kiểm kê tài nguyên

| Loại | Tên / ID | Owner | Ghi chú |
|---|---|---|---|
| Repo git (remote) | `github.com/namnd020304/clone-python-yolo` | namnd020304 | Nhánh `main` |
| Trọng số gốc | `yolov8n.pt` (Ultralytics release v8.3.0) | Ultralytics | Không commit (`*.pt` trong `.gitignore`); tải lại theo `huong-dan/cai-dat-moi-truong.md` |
| Trọng số COCO đổi định dạng | `yolov8n_mine.pt` | Chủ repo | Tạo từ `yolov8n.pt` bằng `train.load_pretrained`; `webcam.py` dùng mặc định |
| Model biển báo (lưu trữ, QĐ-10) | `runs/bien-bao/best.pt` (+ `train.log`) | Chủ repo | Không commit vào git. Bản tải về: [Release v1.0](https://github.com/namnd020304/clone-python-yolo/releases/tag/v1.0), SHA-256 `7fbdc02e…a0ca4ca` |
| Dataset | `../data/coco` (COCO 2017, ~26 GB, ngoài repo), `datasets/coco8` | COCO / Ultralytics | Tải lại được bằng lệnh ở `quy-trinh-train.md` mục 2 |
| Model COCO tự train | `runs/coco-gpu/best.pt` | Chủ repo | Chưa có; khi có thì đưa lên Release mới, ghi SHA-256 ở đây |
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
- [ ] `huong-dan/van-hanh.md` đủ lệnh và tham số; `ban-giao/quy-trinh-train.md` khớp `train.py` hiện tại
- [ ] Bảng kết quả đo có số thật, kèm ngày
- [ ] `runs/bien-bao/best.pt` có trong Release v1.0 (không nằm trong git)

## 3. Người tiếp nhận tự kiểm chứng (chỉ đọc tài liệu, không hỏi)

Bước nào phải hỏi thì sửa tài liệu chỗ đó.

- [ ] Cài môi trường, `python check.py` qua 16/16
- [ ] `python load_weights.py` ra `result.jpg` có 4 người + 1 xe buýt, lệch ≤ 1e-4
- [ ] Theo `ban-giao/quy-trinh-train.md`: chạy thử coco8 (mục 3.1), rồi chạy lệnh COCO mục 3.2 tới khi dòng đầu log
  ghi `thiết bị cuda` và thanh tiến trình đạt ~8 batch/s (không cần chạy hết)
- [ ] `predict.py` trên một ảnh val, box đúng biển
- [ ] `webcam.py` mở camera, có box quanh người

## 4. Buổi chuyển giao

1. Tổng quan và đọc tài liệu theo `docs/README.md` (1 giờ).
2. Người tiếp nhận tự cài và chạy, người cũ chỉ sửa tài liệu (1 giờ).
3. Đi qua code: blocks → model → head (DFL) → TAL → loss → train (1–2 giờ). TAL và loss là phần khó nhất.
4. Người tiếp nhận làm một việc nhỏ trọn vẹn đến commit, ví dụ thêm `--resume` (nửa ngày).
