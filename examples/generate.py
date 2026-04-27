#!/usr/bin/env python3
"""Dataset generation entry point for M-105 surgical-phase-recognition.

Usage:
    python examples/generate.py
    python examples/generate.py --num-samples 10
    python examples/generate.py --output data/my_output
"""
import argparse
import sys
from pathlib import Path

# Live-stream stdout so EC2 log tail sees progress without buffering delay.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TaskPipeline, TaskConfig


def main():
    parser = argparse.ArgumentParser(description="Generate M-105 surgical-phase-recognition dataset")
    parser.add_argument("--num-samples", type=int, default=800)
    parser.add_argument("--output", type=str, default="data/questions")
    args = parser.parse_args()

    print(f"[M-105_pitvis_surgical_phase_recognition] Generating {args.num_samples} sample(s)",
          flush=True)
    config = TaskConfig(
        num_samples=args.num_samples,
        output_dir=Path(args.output),
    )
    pipeline = TaskPipeline(config)
    samples = pipeline.run()
    print(f"[M-105_pitvis_surgical_phase_recognition] Wrote {len(samples)} samples",
          flush=True)


if __name__ == "__main__":
    main()
