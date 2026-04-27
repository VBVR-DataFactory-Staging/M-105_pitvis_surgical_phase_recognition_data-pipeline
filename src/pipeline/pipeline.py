"""Pipeline for M-105 — MedVQA-2019 ImageCLEF clinical visual QA.

Repurposed from the original PitVis scaffold (Figshare blocked AWS IPs and
HF git-LFS kept failing). MedVQA-2019 has a permissive Zenodo direct
download already mirrored to s3://med-vr-datasets/M-105/medvqa2019/.

Each sample = one medical image + 4 QA pairs (Modality / Plane / Organ
System / Abnormality). The ground-truth video sequentially reveals each
question and its answer overlaid next to the image.
"""
from __future__ import annotations
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional

import cv2

from core.pipeline import BasePipeline, OutputWriter, SampleProcessor, TaskSample
from src.download.downloader import create_downloader
from src.pipeline.config import TaskConfig
from src.pipeline.transforms import (
    build_frames,
    load_and_resize,
    make_video,
    render_panel,
    select_qa_pairs,
)


PROMPT = (
    "This video shows a clinical medical image (MedVQA-2019 ImageCLEF) "
    "annotated with a sequence of question/answer panels covering four "
    "diagnostic categories: Modality, Imaging Plane, Organ System, and "
    "Abnormality. Each panel reveals one question (yellow) followed by its "
    "ground-truth answer (green). For each question shown in the video, "
    "report the answer exactly as displayed."
)


TMP_DIR = Path("_tmp")


class TaskPipeline(BasePipeline):
    def __init__(self, config: Optional[TaskConfig] = None):
        super().__init__(config or TaskConfig())
        self.task_config: TaskConfig = self.config  # type: ignore[assignment]
        self.downloader = create_downloader(self.task_config)

    def download(self) -> Iterator[dict]:
        yield from self.downloader.iter_samples()

    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        image_id = raw_sample["image_id"]
        image_path = raw_sample["image_path"]
        qa = raw_sample["qa"]
        cfg = self.task_config

        base = load_and_resize(image_path, tuple(cfg.target_size))
        if base is None:
            return None

        chosen = select_qa_pairs(qa, cfg.max_qa_pairs)
        if not chosen:
            return None

        frames, first_rgb, last_rgb, last_segment = build_frames(
            base_img=base,
            qa_pairs=chosen,
            frames_per_panel=cfg.frames_per_panel,
        )

        # First-segment clip: image-only intro + first Q/A reveal.
        intro_panel = render_panel(base, None, "Reviewing medical image...", None, 0, len(chosen))
        first_q = render_panel(base, chosen[0].get("category"), chosen[0]["question"], None, 0, len(chosen))
        first_a = render_panel(base, chosen[0].get("category"), chosen[0]["question"], chosen[0]["answer"], 0, len(chosen))
        first_segment = (
            [intro_panel] * cfg.frames_per_panel
            + [first_q] * cfg.frames_per_panel
            + [first_a] * cfg.frames_per_panel
        )

        task_id = f"medvqa2019_{image_id}_{idx:05d}"
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        tmp = TMP_DIR / task_id
        tmp.mkdir(parents=True, exist_ok=True)

        gt_path = tmp / "ground_truth.mp4"
        first_path = tmp / "first_video.mp4"
        last_path = tmp / "last_video.mp4"

        make_video(frames, gt_path, cfg.fps)
        make_video(first_segment, first_path, cfg.fps)
        make_video(last_segment, last_path, cfg.fps)

        metadata = {
            "task_id": task_id,
            "source_dataset": "MedVQA-2019 ImageCLEF (Zenodo)",
            "image_id": image_id,
            "split": raw_sample.get("split"),
            "num_qa_pairs": len(chosen),
            "qa_pairs": [
                {
                    "category": q.get("category"),
                    "question": q["question"],
                    "answer": q["answer"],
                }
                for q in chosen
            ],
            "fps": cfg.fps,
        }

        return SampleProcessor.build_sample(
            task_id=task_id,
            domain=cfg.domain,
            first_image=cv2.cvtColor(base, cv2.COLOR_BGR2RGB),
            prompt=PROMPT,
            final_image=last_rgb,
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
        cap = self.task_config.num_samples or 300
        processed = 0
        try:
            for idx, raw in enumerate(self.download()):
                if processed >= cap:
                    print(f"  Hit num_samples cap={cap}, stopping")
                    break
                try:
                    sample = self.process_sample(raw, idx)
                except Exception as e:
                    print(f"  [warn] sample {idx} failed: {e}")
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
                    print(f"  Processed {processed}/{cap} samples")
            print(f"Done! Processed {processed} samples (cap={cap})")
        finally:
            if TMP_DIR.exists():
                shutil.rmtree(TMP_DIR, ignore_errors=True)
        return samples
