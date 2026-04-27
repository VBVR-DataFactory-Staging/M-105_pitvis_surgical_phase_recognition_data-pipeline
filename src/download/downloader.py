"""Downloader for M-105 surgical-phase-recognition.

Pulls the CholecSeg8k mirror via aws s3 sync (NEVER http) and yields one
sample dict per consecutive video segment.

Layout on S3:
    s3://med-vr-datasets/M-080_CholecSeg8k/raw/
        video01/video01_00080/frame_<N>_endo.png ...
        video01/video01_00160/frame_<N>_endo.png ...
        ...
        video12/video12_<NNNNN>/frame_<N>_endo.png ...

Each `video<XX>_<startN>` subdirectory holds ~80 consecutive endoscopic
frames. We treat each subdir as one phase-recognition sample whose phase
label is derived from the segment's relative position within its parent
video (using approximate Cholec80 phase boundaries).
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterator, List, Optional


SEGMENT_RE = re.compile(r"^video(\d+)_(\d+)$")


def _aws_sync(bucket: str, prefix: str, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "sync",
        f"s3://{bucket}/{prefix}",
        str(dst),
        "--only-show-errors",
        "--no-progress",
        "--exclude", "*_color_mask.png",  # don't need color masks
    ]
    print("[download]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _ensure_raw(config) -> Path:
    raw_dir = Path(config.raw_dir)
    if any(raw_dir.glob("video*/video*_*/frame_*_endo.png")):
        print(f"[download] raw already present at {raw_dir}", flush=True)
        return raw_dir
    _aws_sync(config.s3_bucket, config.s3_prefix.rstrip("/"), raw_dir)
    return raw_dir


def _build_index(root: Path) -> List[dict]:
    """Walk all video<NN>/ subdirs, group segments by parent video, compute
    relative position within each parent video, return a flat list of dicts.
    """
    by_video: dict = {}
    for vid_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("video")):
        segs: List[dict] = []
        for seg_dir in sorted(p for p in vid_dir.iterdir() if p.is_dir()):
            m = SEGMENT_RE.match(seg_dir.name)
            if not m:
                continue
            start_frame = int(m.group(2))
            frames = sorted(seg_dir.glob("frame_*_endo.png"))
            if not frames:
                continue
            segs.append({
                "video_id": vid_dir.name,
                "segment_id": seg_dir.name,
                "start_frame": start_frame,
                "frames": frames,
            })
        if segs:
            by_video[vid_dir.name] = segs

    samples: List[dict] = []
    for vid_id, segs in by_video.items():
        segs.sort(key=lambda s: s["start_frame"])
        max_start = max(s["start_frame"] for s in segs) or 1
        for s in segs:
            rel = s["start_frame"] / max_start if max_start > 0 else 0.0
            s["rel_pos"] = rel
            samples.append(s)
    print(f"[download] indexed {len(samples)} segments across {len(by_video)} videos", flush=True)
    return samples


class TaskDownloader:
    def __init__(self, config):
        self.config = config

    def iter_samples(self, limit: Optional[int] = None) -> Iterator[dict]:
        root = _ensure_raw(self.config)
        all_samples = _build_index(root)
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
