"""Deterministic cropping of League HUD regions from frames."""

import cv2
import numpy as np

from extractor.config import get_crop_regions


def crop_hud_regions(
    frame: np.ndarray,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    """Crop all HUD regions from a frame.

    Returns a dict of region_name -> cropped numpy array.
    """
    regions = get_crop_regions(width, height)
    crops = {}

    for name, r in regions.items():
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        # Clamp to frame bounds
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w > 0 and h > 0:
            crops[name] = frame[y:y + h, x:x + w].copy()

    return crops


def save_crops_to_disk(
    crops: dict[str, np.ndarray],
    output_dir: str,
    frame_index: int,
) -> dict[str, str]:
    """Save crop images to disk for debugging. Returns paths dict."""
    paths = {}
    for name, crop in crops.items():
        filename = f"frame_{frame_index:06d}_{name}.png"
        path = f"{output_dir}/{filename}"
        cv2.imwrite(path, crop)
        paths[name] = path
    return paths
