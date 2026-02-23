"""Frame sampling from video files using OpenCV."""

from collections.abc import Iterator

import cv2
import numpy as np

from extractor.config import DEFAULT_SAMPLE_FPS, MAX_FRAMES_PER_EXTRACTION


def sample_frames(
    video_path: str,
    fps: float = DEFAULT_SAMPLE_FPS,
    max_frames: int = MAX_FRAMES_PER_EXTRACTION,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield (frame_index, timestamp_ms, frame) at the target fps.

    Uses OpenCV seek — never loads the entire video into RAM.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0 or total_frames <= 0:
        cap.release()
        return

    step = max(1, int(video_fps / fps))
    frame_count = 0

    for idx in range(0, total_frames, step):
        if frame_count >= max_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        ts_ms = int((idx / video_fps) * 1000)
        yield (idx, ts_ms, frame)
        frame_count += 1

    cap.release()


def get_video_info(video_path: str) -> dict:
    """Return basic video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

    if info["fps"] > 0 and info["total_frames"] > 0:
        info["duration_ms"] = int((info["total_frames"] / info["fps"]) * 1000)
    else:
        info["duration_ms"] = None

    cap.release()
    return info
