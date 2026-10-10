"""Prepare the Oxford-IIIT Pet dataset for this project.

Steps:
1. Download images.tar.gz and annotations.tar.gz from the official source (if missing).
2. Extract them into the raw directory.
3. Use the OFFICIAL split files (annotations/trainval.txt, annotations/test.txt)
   so the train/test split is fixed and reproducible.
4. Convert every image to RGB, resize so the longest side is at most --max-size px,
   and save as JPEG into data/<split>/<breed>/<image>.jpg
5. Write data/labels.csv listing every image with its label and split.

Usage:
    python image_classification/scripts/prepare_data.py --raw-dir <folder for downloads>
"""

import argparse
import csv
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

from PIL import Image

BASE_URL = "https://thor.robots.ox.ac.uk/datasets/pets"
ARCHIVES = ["images.tar.gz", "annotations.tar.gz"]
SPLIT_FILES = {"train": "trainval.txt", "test": "test.txt"}
SPECIES = {"1": "cat", "2": "dog"}


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVES:
        target = raw_dir / name
        if target.exists():
            print(f"[skip] {name} already downloaded")
            continue
        print(f"[download] {BASE_URL}/{name}")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", target)


def extract(raw_dir: Path) -> Path:
    extract_dir = raw_dir / "extracted"
    for name in ARCHIVES:
        folder = extract_dir / name.split(".")[0]
        if folder.exists():
            print(f"[skip] {name} already extracted")
            continue
        print(f"[extract] {name}")
        with tarfile.open(raw_dir / name) as tar:
            tar.extractall(extract_dir, filter="data")
    return extract_dir


def read_split(path: Path) -> list[dict]:
    """Each line: <image_name> <class_id 1-37> <species 1=cat 2=dog> <breed_id>"""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        image_name, class_id, species, _ = line.split()
        rows.append({
            "image_name": image_name,
            "label": image_name.rsplit("_", 1)[0],  # e.g. "Abyssinian_100" -> "Abyssinian"
            "class_id": int(class_id),
            "species": SPECIES[species],
        })
    return rows


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="folder for downloaded/extracted archives (not committed to git)")
    parser.add_argument("--out-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--max-size", type=int, default=320,
                        help="longest image side in pixels after resizing")
    args = parser.parse_args()

    download(args.raw_dir)
    extract_dir = extract(args.raw_dir)
    images_dir = extract_dir / "images"
    ann_dir = extract_dir / "annotations"

    records = []
    for split, filename in SPLIT_FILES.items():
        for row in read_split(ann_dir / filename):
            src = images_dir / f"{row['image_name']}.jpg"
            rel_path = Path(split) / row["label"] / f"{row['image_name']}.jpg"
            dst = args.out_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(src) as img:
                img = img.convert("RGB")  # a few originals are grayscale / RGBA / CMYK
                img.thumbnail((args.max_size, args.max_size))
                img.save(dst, "JPEG", quality=90)

            records.append({"filepath": rel_path.as_posix(), "label": row["label"],
                            "class_id": row["class_id"], "species": row["species"],
                            "split": split})

    labels_csv = args.out_dir / "labels.csv"
    with labels_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "class_id", "species", "split"])
        writer.writeheader()
        writer.writerows(records)

    counts = Counter((r["label"], r["split"]) for r in records)
    labels = sorted({r["label"] for r in records}, key=str.lower)
    print(f"\n{'label':30s} {'train':>6s} {'test':>6s}")
    for label in labels:
        print(f"{label:30s} {counts[(label, 'train')]:6d} {counts[(label, 'test')]:6d}")
    n_train = sum(r["split"] == "train" for r in records)
    print(f"\nTotal: {len(records)} images, {len(labels)} classes "
          f"(train {n_train}, test {len(records) - n_train})")
    print(f"Labels written to {labels_csv}")


if __name__ == "__main__":
    main()
