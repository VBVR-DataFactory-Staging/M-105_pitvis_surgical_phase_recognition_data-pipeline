"""Pipeline configuration for M-105 MedVQA-2019 (ImageCLEF clinical visual QA)."""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """MedVQA-2019 (ImageCLEF) clinical visual QA task config."""

    domain: str = Field(default="medvqa2019_clinical_qa")

    s3_bucket: str = Field(default="med-vr-datasets")
    # raw extracted on S3:
    #   medvqa2019/ImageClef-2019-VQA-Med-Training/Train_images/synpic*.jpg
    #   medvqa2019/ImageClef-2019-VQA-Med-Training/QAPairsByCategory/C{1..4}_*_train.txt
    #   medvqa2019/ImageClef-2019-VQA-Med-Validation/Val_images/synpic*.jpg
    #   medvqa2019/ImageClef-2019-VQA-Med-Validation/QAPairsByCategory/C{1..4}_*_val.txt
    s3_prefix: str = Field(default="M-105/medvqa2019/")
    fps: int = Field(default=4)
    raw_dir: Path = Field(default=Path("raw"))
    target_size: tuple = Field(default=(640, 640))
    # Each sample has up to 4 QA pairs (one per category).
    max_qa_pairs: int = Field(default=4)
    # Frames each (question_only) and (question+answer) panel is held.
    frames_per_panel: int = Field(default=5)
