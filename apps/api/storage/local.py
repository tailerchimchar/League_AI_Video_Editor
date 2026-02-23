"""Local disk storage abstraction. Can be swapped for S3/R2 later."""

import os
import shutil
from pathlib import Path

# Resolve data dir relative to repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = _REPO_ROOT / "data"


class LocalStorage:
    def __init__(self, base_dir: Path | None = None):
        self.base = base_dir or DATA_DIR
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for subdir in ("videos", "frames", "crops", "debug", "models"):
            (self.base / subdir).mkdir(parents=True, exist_ok=True)

    @property
    def videos_dir(self) -> Path:
        return self.base / "videos"

    @property
    def frames_dir(self) -> Path:
        return self.base / "frames"

    @property
    def crops_dir(self) -> Path:
        return self.base / "crops"

    @property
    def debug_dir(self) -> Path:
        return self.base / "debug"

    def video_path(self, video_id: str, ext: str) -> Path:
        return self.videos_dir / f"{video_id}{ext}"

    def frame_dir(self, video_id: str) -> Path:
        d = self.frames_dir / video_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def crop_dir(self, video_id: str) -> Path:
        d = self.crops_dir / video_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_video(self, video_id: str, ext: str, data: bytes) -> Path:
        path = self.video_path(video_id, ext)
        path.write_bytes(data)
        return path

    def delete_video(self, video_id: str, ext: str) -> None:
        path = self.video_path(video_id, ext)
        if path.exists():
            path.unlink()

    def get_relative_path(self, absolute_path: Path) -> str:
        """Return path relative to data dir for DB storage."""
        return str(absolute_path.relative_to(self.base))


storage = LocalStorage()
