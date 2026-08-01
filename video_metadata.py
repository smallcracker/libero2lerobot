"""Helpers for storing LeRobot v2.1 video metadata."""

from pathlib import Path
from typing import Any

try:
    from lerobot.datasets.video_utils import get_video_info
except ModuleNotFoundError:
    from lerobot.common.datasets.video_utils import get_video_info


_VIDEO_INFO_FEATURE_FIELDS = {"video.height", "video.width", "video.channels"}


def get_v21_video_info(video_path: Path) -> dict[str, Any]:
    """Read a video's metadata using LeRobot's v2.1 schema conventions."""

    if not video_path.is_file():
        raise FileNotFoundError(f"Expected video file not found: {video_path}")

    try:
        video_info = get_video_info(video_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to read video metadata from {video_path}") from exc

    if not video_info:
        raise RuntimeError(f"No video stream found in {video_path}")

    return {
        key: value
        for key, value in video_info.items()
        if key not in _VIDEO_INFO_FEATURE_FIELDS
    }
