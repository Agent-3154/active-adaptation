import os
from pathlib import Path
from typing import Optional

import numpy as np
import imageio.v2 as imageio


class VideoRecorder:
    """Base context-manager interface for recording videos from an environment."""

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def add_frame(self):
        """Record a single frame from the environment."""
        raise NotImplementedError

    def close(self):
        """Finalize and save the video if needed."""
        raise NotImplementedError


class NullVideoRecorder(VideoRecorder):
    """No-op recorder used when recording is disabled or unsupported."""

    def add_frame(self):
        return None

    def close(self):
        return None


class IsaacVideoRecorder(VideoRecorder):
    """Video recorder for Isaac backend using Replicator rgb annotator.

    Frames are buffered in memory and written on context exit so that videos are
    still saved when the loop is interrupted with KeyboardInterrupt.
    """

    def __init__(self, env, path: str | os.PathLike, enabled: bool = True, fps: Optional[int] = None):
        # Only Isaac backend with replicator is expected to support rgb_array.
        if not hasattr(env, "_rgb_annotator"):
            raise ValueError("Environment does not have a `_rgb_annotator` for rgb_array rendering.")
        self._env = env
        self._enabled = bool(enabled)
        self._path = Path(path)
        self._fps = fps or int(1.0 / float(getattr(env, "step_dt", 1.0)))
        self._writer: Optional[imageio.Writer] = None

        if self._enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Open an ffmpeg-backed writer and stream frames as they arrive.
            self._writer = imageio.get_writer(
                str(self._path),
                fps=self._fps,
                codec="h264",
            )

    def add_frame(self):
        if not self._enabled or self._writer is None:
            return
        frame = self._env.render("rgb_array")
        # Ensure we have an H x W x 3 uint8 array on CPU.
        frame_np = np.asarray(frame, dtype=np.uint8)
        self._writer.append_data(frame_np)

    def close(self):
        if not self._enabled or self._writer is None:
            return
        # Finalize container and flush to disk. If the process is killed abruptly
        # some trailing frames may be lost, but already-written data is kept.
        self._writer.close()
        self._writer = None
        print(f"Video saved to: {self._path}")

