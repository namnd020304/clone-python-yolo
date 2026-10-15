# Nhật ký công việc

Mỗi buổi làm một mục, mới nhất ở trên. Chỉ ghi điều code không cho thấy: đã thử gì, số đo, vì sao đổi hướng. Không
chép lại diff. Quyết định thiết kế thì ghi vào [../kien-truc/quyet-dinh.md](../kien-truc/quyet-dinh.md).

Mẫu: `## dd/mm/yyyy`, rồi các gạch đầu dòng: Đã làm / Kết quả / Vấn đề / Tiếp theo.

## 15/10/2026 (chiều): train COCO từ số 0 trên GCP

- Tải COCO 2017 lên VM `yolo-coco` (`n2-standard-8`, `us-east1-b`, ổ 150 GB; us-central1 hết máy): chia như
  `coco.yaml` của Ultralytics. train2017 118.287 ảnh (117.266 file nhãn), val2017 5.000 (4.952), test2017 40.670
  (không có nhãn, chỉ chấm được qua server COCO). Tải ~25 GB mất ~2 phút.
- Đo trên VM: train 0,146 s/ảnh (batch 8 hay 16 như nhau) → **~4,7 giờ/epoch**; đo mAP cả val2017 ~3,6 phút. Tiến
  trình train ~380% CPU, worker đọc ảnh gần như rảnh: nút thắt là tính toán, không phải mosaic.
- Thêm `--resume` (last.pt ghi mỗi epoch, kèm optimizer, EMA, vị trí lr) và `--log-every` cho `train.py`.
- Chạy 30 epoch từ 05:06 UTC 15/10, `--lr 0.001` (lr tự tính 0.000119 là quy tắc Ultralytics cho job ngắn; job dài
  họ dùng SGD 0.01), `--eval-every 1`, log ở `~/yolo/runs/coco-scratch/train.log`. Dự kiến xong ~21/10, ~56 USD
  credit; VM tự tắt khi xong. Loss vòng 500 → 4000 của epoch 1: box 3.63 → 3.08, cls 5.70 → 4.91, dfl 4.26 → 3.41.
- Viết lại `load_mosaic`: giữ thuật toán mosaic của YOLOv8 nhưng trình bày khác bản Ultralytics (hướng (hx, hy) thay
  cho 4 nhánh if; đổi toạ độ thẳng sang ảnh đầu ra, mỗi ảnh resize một lần). 2.000 mẫu: không box lỗi, 7,0 box/ảnh
  (bản cũ 7,2), 30 ms/ảnh (bản cũ 35). **Job COCO đang chạy vẫn dùng bản mosaic cũ.**
- Lỗi đã gặp: `wait` trong `prepare.sh` đợi luôn tiến trình `tee` của `exec > >(tee ...)` nên treo ~35 phút sau khi
  tải xong. Không dùng `exec > >(tee)` cùng `wait` không đối số.
- Tiếp theo: theo dõi mAP từng epoch; lấy `last.pt` về máy định kỳ; xong thì lấy `runs/` về và **xoá VM**.

## 15/10/2026 (sáng)

- Dựng bộ tài liệu bàn giao trong `docs/` theo mẫu: `README`, `ke-hoach-hoc.md` (plan học lấy từ README gốc và các
  buổi trước), `ban-giao/`, `kien-truc/`, `huong-dan/`. Bộ lý thuyết `docs/train/` giữ nguyên.
- Vấn đề: trong terminal của trợ lý AI, Windows chặn `torch_python.dll` của `venv` ("An Application Control policy has
  blocked this file"), nên buổi này không chạy lại được `check.py`. Không ảnh hưởng terminal của người dùng.
- Commit `0fa7f9d` (mosaic + EMA, `webcam.py`, tài liệu bàn giao, báo cáo) và push lên `main`. Repo vẫn public.
- Tạo [Release v1.0](https://github.com/namnd020304/clone-python-yolo/releases/tag/v1.0) để chứa `best.pt` (12 MB,
  không nằm trong git) và `train.log`.
- Tiếp theo: bàn giao cho leader theo `checklist-ban-giao.md` (chọn thêm collaborator hay transfer repo).

## 12/10/2026

- Lấy kết quả train qua đêm trên GCP: 100 epoch xong không lỗi, mỗi epoch ~100 s, tổng ~2 giờ 50 phút, chi phí ước
  tính ~1,2 USD credit dùng thử (theo đơn giá nhớ lại, chưa đối chiếu hoá đơn). Đã xoá VM và ổ đĩa.
- mAP theo epoch: 0 → 0.019; 10 → 0.598; 25 → 0.685; 50 → 0.702; 75 → **0.705** (best); 100 → 0.700. Từ epoch 25
  mAP gần như đứng yên trong khi loss box vẫn giảm (0.48 → 0.08): model bắt đầu học thuộc tập train. 30–50 epoch là đủ.
- Thử `predict.py` trên 3 ảnh val: `road160` đúng `stop 0.98`, `crosswalk 0.97` nhưng bỏ sót một biển crosswalk
  ~7×9 px ở xa; `road108` đúng `speedlimit` 13×12 px (0.89); `road239` đúng nhưng điểm chỉ 0.53. **Vật nhỏ là điểm yếu.**
- Thêm mosaic augmentation và EMA (bước 16), bài chấm 16. Vẽ ảnh mosaic kèm nhãn ra `runs/mosaic_check.jpg`: box khớp
  vật. Bài chấm EMA lần đầu viết vòng vo gần như tự đúng; viết lại để so với công thức tính tay.
- Phép thử học thuộc với EMA: 50 → 0.000, 100 → 0.000, 150 → 0.096, 200 → 0.990, 300 → 0.995 (không EMA: 0 / 0 /
  0.202 / 0.995 / 0.995). EMA chạy đúng, nhưng **không** sửa được hiện tượng mAP = 0 tới epoch 100, nên giả thuyết
  "thống kê BatchNorm theo không kịp" không được ủng hộ (van-de-mo #2).
- Viết `webcam.py`: mặc định chỉ hiện `person`, model COCO `yolov8n_mine.pt`. Đo trên CPU: 206 / 170 / 105 ms mỗi khung
  ở size 640 / 480 / 320. Size 320 tìm được 3/4 người trên `bus.jpg` (size 640 tìm đủ 4).
- Viết báo cáo toàn bộ quy trình (bản Docs, rồi xuất `docs/bao-cao-yolov8n.docx` bằng thư viện `docx` của Node vì máy
  không có pandoc). Đo lại để ghi đúng: BatchNorm `eps=1e-5` làm đầu ra lệch tới 141, `eps=1e-3` chỉ lệch 6,1e-05.
- Commit `c06cde6`: dataset, TAL, loss, mAP, `train.py`, `predict.py` (chưa gồm bước 16 và `webcam.py`).

## 11/10/2026

- Cài torch vào `venv` (Python 3.10). Sửa 2 chỗ trong `blocks.py`: bỏ điều kiện thừa `c == c` trong `Bottleneck`;
  `SPPF` padding `2` → `k // 2`. Đổi BatchNorm sang `eps=1e-3, momentum=0.03` giống Ultralytics (QĐ-03).
- Viết `load_weights.py`: tải `yolov8n.pt` từ GitHub release của Ultralytics (v8.3.0), đổi tên khoá
  `state_dict` bằng bảng `LAYERS` / `PARTS` (QĐ-04). 354 tensor nạp vừa khít, đầu ra lệch 6,1e-05. `bus.jpg`: 4 người + 1 xe buýt.
- Viết phần train (bước 10–15) và mở rộng `check.py` lên 15 bài. Loss khớp Ultralytics tuyệt đối ngay lần so đầu.
- Lỗi: lr = 0.002 phá trọng số gốc sau vài bước (mAP về 0 dù loss vẫn giảm). Đổi lr mặc định thành công thức của
  Ultralytics `0.002 × 5 / (4 + nc)` (≈ 0.00012 với 80 class) (QĐ-06).
- Viết `kaggle_data.py`; thử ẩn danh (không cần tài khoản) với `andrewmvd/road-sign-detection` (VOC). Kiểm tay một
  box: khớp. Thử thêm kiểu Roboflow dựng lại từ coco8: nhãn khớp từng số, đọc đúng 80 tên class. `train.py` tự đọc
  `classes.txt`.
- Một lệnh `sed` chèn nhầm ký tự xuống dòng thật vào code; đã sửa tay. Bài học: sửa nhiều dòng thì dùng editor.
- Fine-tune biển báo 3 epoch trên CPU máy nhà: ~2,5–3 phút/epoch. Đưa lên máy ảo GCP (CPU, vì tài khoản free trial
  không có hạn mức GPU): `n2-standard-8`, `us-central1-a`, chạy trong `tmux`, tự tắt khi xong (QĐ-08).
  `plink` (SSH trên Windows) lỗi với lệnh nhiều dòng nên gửi một file script lên rồi chạy; `pip` của Debian quá cũ
  nên nâng `pip` trước.
- Viết bộ lý thuyết `docs/train/` (01–07). Mọi ví dụ số được tính lại bằng code của repo hoặc lấy từ log thật.
- Viết tài liệu "Tự dựng lại YOLOv8n từ số 0" (bản Docs trên claude.ai) cho giai đoạn 1.

## 09/10/2026

- Viết xong bước 1–9 (neck, head có DFL, letterbox, NMS, postprocess). Commit `562433f`.
