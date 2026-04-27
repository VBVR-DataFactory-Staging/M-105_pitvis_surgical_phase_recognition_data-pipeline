"""Frame/video transforms for M-105 surgical-phase-recognition.

Renders each raw endoscopic frame with a phase-labelled banner underneath
(coloured by phase id) and stacks them into an MP4 clip. NO horizontal flip
— surgical anatomy is asymmetric.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


# ── Cholec80 phase taxonomy (BGR colours for OpenCV) ──────────────────────
# Canonical 7-phase Cholec80 ordering (Twinanda et al., 2017):
#   P0 Preparation
#   P1 CalotTriangleDissection
#   P2 ClippingCutting
#   P3 GallbladderDissection
#   P4 GallbladderPackaging
#   P5 CleaningCoagulation
#   P6 GallbladderRetraction
PHASES: List[Tuple[str, Tuple[int, int, int]]] = [
    ("Preparation",              ( 80, 180, 255)),  # warm orange
    ("CalotTriangleDissection",  ( 70, 200,  90)),  # green
    ("ClippingCutting",          (220, 100, 220)),  # magenta
    ("GallbladderDissection",    (200, 200,  60)),  # cyan-ish
    ("GallbladderPackaging",     ( 80, 130, 240)),  # red-orange
    ("CleaningCoagulation",      (210, 170,  80)),  # blue
    ("GallbladderRetraction",    (160, 110, 220)),  # violet
]


def phase_for_segment(rel_pos: float) -> int:
    """Map a segment's relative position within its parent video (0..1) to a
    Cholec80 phase id using literature-derived approximate boundaries.

    Boundaries (cumulative frame fraction in Cholec80, averaged across the
    public training videos): 0.05 / 0.30 / 0.35 / 0.70 / 0.78 / 0.92 / 1.00.
    """
    if rel_pos < 0.05:
        return 0
    if rel_pos < 0.30:
        return 1
    if rel_pos < 0.35:
        return 2
    if rel_pos < 0.70:
        return 3
    if rel_pos < 0.78:
        return 4
    if rel_pos < 0.92:
        return 5
    return 6


def load_frame(path: Path, size: Tuple[int, int]) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def render_with_banner(
    frame: np.ndarray,
    phase_idx: int,
    phase_total: int,
    show_label: bool,
    banner_height: int,
) -> np.ndarray:
    """Compose a frame with a phase-coloured banner underneath.

    If ``show_label`` is False, the banner is rendered solid-coloured with
    "Phase: ?" (used for the "first_video" pre-reveal segment).
    """
    h, w = frame.shape[:2]
    canvas = np.zeros((h + banner_height, w, 3), dtype=np.uint8)
    canvas[:h] = frame

    name, color = PHASES[phase_idx]
    canvas[h:] = color

    cv2.line(canvas, (0, h), (w, h), (20, 20, 20), 3)

    header = f"Phase {phase_idx + 1}/{phase_total}"
    cv2.putText(canvas, header, (24, h + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

    label = name if show_label else "?"
    scale = 1.0 if len(label) < 22 else 0.78
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    tx = max(24, (w - tw) // 2)
    ty = h + banner_height - 28
    cv2.putText(canvas, "Phase:", (tx - 110, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def make_video(frames: Iterable[np.ndarray], out_path: Path, fps: int) -> None:
    frames = list(frames)
    if not frames:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w2, h2 = w - (w % 2), h - (h % 2)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", f"scale={w2}:{h2}",
        str(out_path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait(timeout=180)
