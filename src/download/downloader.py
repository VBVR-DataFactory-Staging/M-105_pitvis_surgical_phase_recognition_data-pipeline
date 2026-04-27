"""Downloader for M-105 — MedVQA-2019 ImageCLEF clinical visual QA.

Lives on S3 under:
    s3://med-vr-datasets/M-105/medvqa2019/
        ImageClef-2019-VQA-Med-Training/
            Train_images/synpic*.jpg                   (~3200 images)
            QAPairsByCategory/C1_Modality_train.txt    (image_id|q|a, one per line)
            QAPairsByCategory/C2_Plane_train.txt
            QAPairsByCategory/C3_Organ_train.txt
            QAPairsByCategory/C4_Abnormality_train.txt
        ImageClef-2019-VQA-Med-Validation/
            Val_images/synpic*.jpg                     (~500 images)
            QAPairsByCategory/C1_Modality_val.txt
            ... (C2/C3/C4)

Each image has up to 4 QA pairs — one from each category.
Yields one dict per image with all matching QA pairs attached.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional


CATEGORIES = [
    ("modality",    "C1_Modality"),
    ("plane",       "C2_Plane"),
    ("organ",       "C3_Organ"),
    ("abnormality", "C4_Abnormality"),
]

SPLITS = [
    # (split_dir, qa_suffix, images_subdir)
    ("ImageClef-2019-VQA-Med-Training",   "train", "Train_images"),
    ("ImageClef-2019-VQA-Med-Validation", "val",   "Val_images"),
]


def _aws_sync(bucket: str, prefix: str, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "sync",
        f"s3://{bucket}/{prefix}",
        str(dst),
        "--only-show-errors",
        "--no-progress",
    ]
    print("[download]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _ensure_raw(config) -> Path:
    """Make sure the medvqa2019/ tree is on local disk; return the root."""
    raw_dir = Path(config.raw_dir)
    # When invoked under the v3 bootstrap, raw/ may already be populated by
    # the pre-sync step (`aws s3 sync s3://.../M-105/medvqa2019/ raw/`). In
    # that case the tree is rooted directly at raw/. Detect by file presence.
    if any(raw_dir.glob("ImageClef-2019-VQA-Med-Training/Train_images/*.jpg")):
        print(f"[download] raw already present at {raw_dir}")
        return raw_dir

    candidate = raw_dir / "medvqa2019"
    if any(candidate.glob("ImageClef-2019-VQA-Med-Training/Train_images/*.jpg")):
        print(f"[download] raw already present at {candidate}")
        return candidate

    # Sync the medvqa2019/ subtree directly into raw/.
    _aws_sync(config.s3_bucket, config.s3_prefix.rstrip("/"), raw_dir)
    return raw_dir


def _parse_qa_file(path: Path, category_label: str) -> dict:
    """Parse `image_id|question|answer` → {image_id: {category, question, answer}}."""
    out: dict = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        image_id, question = parts[0].strip(), parts[1].strip()
        # Some answers contain "|", rejoin.
        answer = "|".join(parts[2:]).strip() if len(parts) > 3 else parts[2].strip()
        if not image_id or not question:
            continue
        out[image_id] = {
            "category": category_label,
            "question": question,
            "answer": answer,
        }
    return out


def _build_index(root: Path) -> List[dict]:
    """Walk both splits and build a flat list of {image_id, image_path, qa: [...]}."""
    samples: List[dict] = []
    for split_dir, qa_suffix, images_subdir in SPLITS:
        split_root = root / split_dir
        if not split_root.exists():
            print(f"[download] skip missing split {split_root}")
            continue
        img_root = split_root / images_subdir
        if not img_root.exists():
            print(f"[download] skip missing images {img_root}")
            continue

        png_index = {p.stem: p for p in img_root.glob("*.jpg")}
        if not png_index:
            print(f"[download] no jpgs in {img_root}")
            continue

        per_category: dict = {}
        for cat_label, cat_prefix in CATEGORIES:
            qa_path = split_root / "QAPairsByCategory" / f"{cat_prefix}_{qa_suffix}.txt"
            per_category[cat_label] = _parse_qa_file(qa_path, cat_label)

        added = 0
        for image_id in sorted(png_index.keys()):
            qa: List[dict] = []
            for cat_label, _ in CATEGORIES:
                rec = per_category.get(cat_label, {}).get(image_id)
                if rec is not None:
                    qa.append(rec)
            if not qa:
                continue
            samples.append({
                "image_id": image_id,
                "image_path": str(png_index[image_id]),
                "split": qa_suffix,
                "qa": qa,
            })
            added += 1
        print(f"[download] split={qa_suffix}: {added} images with QA")
    return samples


class TaskDownloader:
    def __init__(self, config):
        self.config = config

    def iter_samples(self, limit: Optional[int] = None) -> Iterator[dict]:
        root = _ensure_raw(self.config)
        all_samples = _build_index(root)
        print(f"[download] total samples indexed: {len(all_samples)}")
        emitted = 0
        for s in all_samples:
            yield s
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    def download(self, limit: Optional[int] = None):
        yield from self.iter_samples(limit=limit)


def create_downloader(config) -> TaskDownloader:
    return TaskDownloader(config)
