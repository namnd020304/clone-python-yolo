"""Tải dataset phát hiện vật thể từ Kaggle và chuyển về cấu trúc train.py dùng:

    datasets/<tên>/images/{train,val}/*.jpg
    datasets/<tên>/labels/{train,val}/*.txt     class cx cy w h (0..1)
    datasets/<tên>/classes.txt                  mỗi dòng một tên class, dòng i = class i

    python -m pip install kagglehub
    python kaggle_data.py andrewmvd/road-sign-detection          # mã dataset = phần cuối URL kaggle.com/datasets/...
    python kaggle_data.py andrewmvd/road-sign-detection --name bien-bao --val 0.2
    python kaggle_data.py D:/thu_muc_da_tai                       # hoặc một thư mục đã có trên máy

Nhận ra 2 định dạng nhãn phổ biến:
  - YOLO: file .txt cạnh ảnh, hoặc ở thư mục "labels" song song với "images" (kiểu Roboflow). Tên class lấy từ
    data.yaml / classes.txt / obj.names nếu có.
  - Pascal VOC: file .xml có <object><name> và <bndbox>. Tên class lấy từ các file xml, xếp theo ABC.
Nếu dataset đã chia sẵn thư mục train / val (valid) / test thì giữ nguyên cách chia (test gộp vào val khi không có
val); không thì chia ngẫu nhiên theo --val.

Dataset công khai tải được không cần tài khoản. Dataset cần đăng nhập: tạo API token ở kaggle.com/settings, đặt file
kaggle.json vào C:/Users/<bạn>/.kaggle/ (hoặc đặt biến môi trường KAGGLE_USERNAME, KAGGLE_KEY).
"""
import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
               "test": "test", "testing": "test"}


def download(source: str) -> Path:
    if Path(source).exists():
        return Path(source)
    try:
        import kagglehub
    except ImportError:
        sys.exit("Cần cài kagglehub: python -m pip install kagglehub")
    print(f"Đang tải {source} từ Kaggle ...")
    return Path(kagglehub.dataset_download(source))


def split_of(path: Path, root: Path) -> str | None:
    """Tên split (train / val / test) nếu một thư mục cha có tên như vậy."""
    for part in path.relative_to(root).parts[:-1]:
        if part.lower() in SPLIT_NAMES:
            return SPLIT_NAMES[part.lower()]
    return None


# ---------- YOLO ----------

def yolo_label_for(img: Path) -> Path | None:
    """images/x.jpg -> labels/x.txt (thư mục "images" gần nhất đổi thành "labels"), hoặc x.txt cạnh ảnh."""
    parts = list(img.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].lower() == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            if cand.exists():
                return cand
    cand = img.with_suffix(".txt")
    return cand if cand.exists() else None


def read_yolo_names(root: Path) -> list[str] | None:
    for p in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
        try:
            import yaml
            names = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("names")
        except Exception:
            continue
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=int)]
        if isinstance(names, list):
            return [str(n) for n in names]
    for fname in ("classes.txt", "obj.names", "_darknet.labels"):
        for p in root.rglob(fname):
            return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return None


def collect_yolo(root: Path, images: list[Path]):
    items = []
    for img in images:
        lab = yolo_label_for(img)
        if lab is None:
            continue
        rows = []
        for line in lab.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) == 5:  # dòng nhiều cột hơn là đa giác segmentation, bỏ qua
                rows.append((int(float(v[0])), *map(float, v[1:5])))
        items.append((img, rows))
    nc = max((r[0] for _, rows in items for r in rows), default=-1) + 1
    names = read_yolo_names(root) or [str(i) for i in range(nc)]
    if len(names) < nc:
        names += [str(i) for i in range(len(names), nc)]
    return items, names


# ---------- Pascal VOC ----------

def collect_voc(root: Path, images: list[Path], xmls: list[Path]):
    by_stem = {}
    for img in images:
        by_stem.setdefault(img.stem, img)
    parsed, names = [], set()
    for x in xmls:
        try:
            ann = ET.parse(x).getroot()
        except ET.ParseError:
            continue
        fname = ann.findtext("filename") or ""
        img = x.with_name(fname) if fname and x.with_name(fname).exists() else by_stem.get(Path(fname).stem or x.stem)
        img = img or by_stem.get(x.stem)
        size = ann.find("size")
        if img is None or size is None:
            continue
        w, h = float(size.findtext("width") or 0), float(size.findtext("height") or 0)
        if w <= 0 or h <= 0:
            from PIL import Image
            w, h = Image.open(img).size
        objs = []
        for o in ann.iter("object"):
            bb = o.find("bndbox")
            if bb is None:
                continue
            x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            name = (o.findtext("name") or "").strip()
            names.add(name)
            objs.append((name, x1, y1, x2, y2))
        parsed.append((img, w, h, objs))
    names = sorted(names)
    idx = {n: i for i, n in enumerate(names)}
    items = []
    for img, w, h, objs in parsed:
        rows = []
        for name, x1, y1, x2, y2 in objs:
            x1, x2 = max(0.0, min(x1, x2)), min(w, max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(h, max(y1, y2))
            if x2 > x1 and y2 > y1:
                rows.append((idx[name], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
        items.append((img, rows))
    return items, names


# ---------- ghi ra ----------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="mã dataset Kaggle owner/ten-dataset, hoặc thư mục trên máy")
    ap.add_argument("--name", help="tên thư mục trong datasets/ (mặc định = phần sau dấu /)")
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--val", type=float, default=0.2, help="tỉ lệ ảnh val khi dataset chưa chia sẵn")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = download(args.source)
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
    xmls = sorted(root.rglob("*.xml"))
    if not images:
        sys.exit(f"Không thấy ảnh nào trong {root}")

    items, names = collect_yolo(root, images)
    fmt = "YOLO"
    if not items and xmls:
        items, names = collect_voc(root, images, xmls)
        fmt = "Pascal VOC"
    if not items:
        sys.exit("Không nhận ra định dạng nhãn (cần YOLO .txt hoặc Pascal VOC .xml). "
                 "Dataset dạng COCO JSON hay CSV chưa được hỗ trợ.")

    # chia train / val
    splits = {img: split_of(img, root) for img, _ in items}
    has_val = "val" in splits.values()
    if "train" in splits.values():
        # giữ cách chia sẵn; test chỉ dùng làm val khi không có val; ảnh không nằm trong split nào -> train
        pick = {"train": "train", "val": "val", "test": None if has_val else "val", None: "train"}
        assign = {img: pick[s] for img, s in splits.items()}
    else:
        rng = random.Random(args.seed)
        order = [img for img, _ in items]
        rng.shuffle(order)
        n_val = max(1, round(len(order) * args.val))
        assign = {img: ("val" if i < n_val else "train") for i, img in enumerate(order)}

    name = args.name or args.source.replace("\\", "/").rstrip("/").split("/")[-1]
    out = Path(args.out) / name
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)

    used, counts, n_boxes = set(), {"train": 0, "val": 0}, 0
    for img, rows in items:
        s = assign.get(img)
        if s is None:
            continue
        stem = img.stem
        while (s, stem) in used:  # trùng tên ở các thư mục con khác nhau
            stem += "_"
        used.add((s, stem))
        shutil.copy2(img, out / "images" / s / f"{stem}{img.suffix.lower()}")
        (out / "labels" / s / f"{stem}.txt").write_text(
            "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows),
            encoding="utf-8", newline="\n")
        counts[s] += 1
        n_boxes += len(rows)
    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8", newline="\n")

    print(f"Định dạng {fmt}: {counts['train']} ảnh train, {counts['val']} ảnh val, {n_boxes} box, {len(names)} class")
    print("Class: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)))
    print(f"Đã ghi vào {out}")
    print(f"\nTrain:  python train.py --data {out.as_posix()} --weights yolov8n.pt")


if __name__ == "__main__":
    main()
