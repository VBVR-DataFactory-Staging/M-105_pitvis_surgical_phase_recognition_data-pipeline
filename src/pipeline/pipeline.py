"""Pipeline for M-105 — surgical-phase-recognition (laparoscopic cholecystectomy).

Each raw segment (~80 consecutive endoscopic frames) becomes one VBVR sample.
The first / last / ground-truth videos are constructed as follows:

  first_video.mp4    : segment frames + question banner ("Phase: ?")
  ground_truth.mp4   : segment frames + colour-coded banner with the
                       Cholec80 phase name revealed
  last_video.mp4     : tail of the segment with the answer banner shown

Augmentation (modest) is OPTIONAL via num_samples > unique-segments. We do not
horizontal-flip — surgical anatomy is asymmetric (gallbladder is right-of-
midline, cystic duct/artery are right-anterior).
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from core.pipeline import BasePipeline, OutputWriter, SampleProcessor, TaskSample
from src.download.downloader import create_downloader
from src.pipeline.config import TaskConfig
from src.pipeline.transforms import (
    PHASES,
    load_frame,
    make_video,
    phase_for_segment,
    render_with_banner,
)


# Live-stream stdout so EC2 log tail sees progress without buffering.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


PROMPT = (
    "This video shows a continuous segment of a laparoscopic cholecystectomy "
    "(gallbladder removal). Identify which of the seven canonical Cholec80 "
    "surgical phases is being performed: Preparation, "
    "CalotTriangleDissection, ClippingCutting, GallbladderDissection, "
    "GallbladderPackaging, CleaningCoagulation, or GallbladderRetraction. "
    "Reveal the phase name in the coloured banner beneath the surgical view, "
    "matching the colour and label shown in the ground-truth video."
)


TMP_DIR = Path("_tmp")


class TaskPipeline(BasePipeline):
    def __init__(self, config: Optional[TaskConfig] = None):
        super().__init__(config or TaskConfig())
        self.task_config: TaskConfig = self.config  # type: ignore[assignment]
        self.downloader = create_downloader(self.task_config)

    def download(self) -> Iterator[dict]:
        yield from self.downloader.iter_samples()

    def _load_segment_frames(self, raw_sample: dict) -> List[np.ndarray]:
        cfg = self.task_config
        size = tuple(cfg.target_size)
        stride = max(1, int(cfg.frame_stride))
        out: List[np.ndarray] = []
        for fp in raw_sample["frames"][::stride]:
            img = load_frame(fp, size)
            if img is not None:
                out.append(img)
        return out

    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        cfg = self.task_config
        seg_id = raw_sample["segment_id"]
        video_id = raw_sample["video_id"]
        rel_pos = raw_sample["rel_pos"]
        phase_idx = phase_for_segment(rel_pos)
        phase_name, phase_color = PHASES[phase_idx]

        frames = self._load_segment_frames(raw_sample)
        if not frames:
            print(f"  [skip] {seg_id}: no readable frames", flush=True)
            return None

        # Build the three video tracks.
        gt_frames = [
            render_with_banner(f, phase_idx, len(PHASES), True, cfg.banner_height)
            for f in frames
        ]
        first_frames = [
            render_with_banner(f, phase_idx, len(PHASES), False, cfg.banner_height)
            for f in frames
        ]
        # Last clip = back half of the segment with answer revealed.
        half = max(1, len(gt_frames) // 2)
        last_frames = gt_frames[half:] or gt_frames[-1:]

        task_id = f"{cfg.domain}_{video_id}_{raw_sample['start_frame']:05d}_{idx:05d}"
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        tmp = TMP_DIR / task_id
        tmp.mkdir(parents=True, exist_ok=True)

        gt_path = tmp / "ground_truth.mp4"
        first_path = tmp / "first_video.mp4"
        last_path = tmp / "last_video.mp4"

        make_video(gt_frames, gt_path, cfg.fps)
        make_video(first_frames, first_path, cfg.fps)
        make_video(last_frames, last_path, cfg.fps)

        first_rgb = cv2.cvtColor(first_frames[0], cv2.COLOR_BGR2RGB)
        final_rgb = cv2.cvtColor(gt_frames[-1], cv2.COLOR_BGR2RGB)

        metadata = {
            "task_id": task_id,
            "source_dataset": "CholecSeg8k (Cholec80 mirror)",
            "video_id": video_id,
            "segment_id": seg_id,
            "start_frame": raw_sample["start_frame"],
            "num_frames": len(frames),
            "fps": cfg.fps,
            "phase_idx": phase_idx,
            "phase_name": phase_name,
            "phase_color_bgr": list(phase_color),
            "rel_pos_in_video": round(rel_pos, 4),
        }

        return SampleProcessor.build_sample(
            task_id=task_id,
            domain=cfg.domain,
            first_image=first_rgb,
            prompt=PROMPT,
            final_image=final_rgb,
            first_video=str(first_path),
            last_video=str(last_path),
            ground_truth_video=str(gt_path),
            metadata=metadata,
        )

    def run(self) -> List[TaskSample]:
        """Incremental-S3-upload run loop (mirrors M-25 / M-102 pattern)."""
        s3_bucket = os.environ.get("INCREMENTAL_S3_BUCKET", "")
        s3_prefix = os.environ.get("INCREMENTAL_S3_PREFIX", "")
        writer = OutputWriter(self.config.output_dir)
        samples: List[TaskSample] = []
        cap = self.task_config.num_samples or 800
        processed = 0
        try:
            for idx, raw in enumerate(self.download()):
                if processed >= cap:
                    print(f"  Hit num_samples cap={cap}, stopping", flush=True)
                    break
                try:
                    sample = self.process_sample(raw, idx)
                except Exception as e:
                    print(f"  [warn] sample {idx} failed: {e}", flush=True)
                    sample = None
                if sample is None:
                    continue
                sample_dir = writer.write_sample(sample)
                samples.append(sample)
                processed += 1
                if s3_bucket and sample_dir is not None:
                    task_dir_name = sample_dir.parent.name
                    sample_name = sample_dir.name
                    dst = f"s3://{s3_bucket}/{s3_prefix}/{task_dir_name}/{sample_name}/"
                    subprocess.run(
                        ["aws", "s3", "cp", "--recursive", str(sample_dir), dst, "--only-show-errors"],
                        check=False,
                    )
                if processed % 10 == 0:
                    print(f"  Processed {processed}/{cap} samples", flush=True)
            print(f"Done! Processed {processed} samples (cap={cap})", flush=True)
        finally:
            if TMP_DIR.exists():
                shutil.rmtree(TMP_DIR, ignore_errors=True)
        return samples
