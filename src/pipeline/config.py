"""Pipeline configuration for M-105 — Surgical Phase Recognition.

Original target was PitVis-2023 (UCL-WEISS pituitary surgery videos), but the
upstream sources (Figshare + HuggingFace git-LFS) are unreachable from the
ingest EC2s. We pivot to laparoscopic cholecystectomy phase recognition using
the CholecSeg8k mirror that already lives on S3 — same surgical-video / phase
recognition task family, well-known Cholec80 phase labels, asymmetric anatomy.

Each raw "video<NN>/video<NN>_<frameN>/" directory becomes one VBVR sample:
the consecutive `frame_*_endo.png` frames are stacked into a short clip, and
the phase label is rendered as a banner overlay on the ground-truth video.
"""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Surgical-phase-recognition task config (CholecSeg8k mirror)."""

    domain: str = Field(default="pitvis_surgical_phase_recognition")

    s3_bucket: str = Field(default="med-vr-datasets")
    # CholecSeg8k mirror — surgical video segments with per-frame masks.
    s3_prefix: str = Field(default="M-080_CholecSeg8k/raw/")
    fps: int = Field(default=8)
    raw_dir: Path = Field(default=Path("raw"))
    target_size: tuple = Field(default=(960, 540))
    # Banner height (px) appended below the surgical frame for the phase label.
    banner_height: int = Field(default=120)
    # Frame stride within a segment (segments are ~80 consecutive frames).
    frame_stride: int = Field(default=2)
