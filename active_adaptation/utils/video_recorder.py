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


class RgbArrayVideoRecorder(VideoRecorder):
    """Stream frames from any backend that supports ``env.render("rgb_array")``."""

    def __init__(self, env, path: str | os.PathLike, enabled: bool = True, fps: Optional[int] = None):
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
        if frame is None:
            raise ValueError("Environment returned no frame for rgb_array rendering.")
        frame_np = np.asarray(frame)
        if frame_np.ndim != 3:
            raise ValueError(
                f"Expected an HxWxC frame from rgb_array rendering, got shape {frame_np.shape!r}."
            )
        if frame_np.shape[-1] == 4:
            frame_np = frame_np[..., :3]
        frame_np = np.asarray(frame_np, dtype=np.uint8)
        self._writer.append_data(frame_np)

    def close(self):
        if not self._enabled or self._writer is None:
            return
        # Finalize container and flush to disk. If the process is killed abruptly
        # some trailing frames may be lost, but already-written data is kept.
        self._writer.close()
        self._writer = None
        print(f"Video saved to: {self._path}")


class IsaacVideoRecorder(RgbArrayVideoRecorder):
    """Backward-compatible alias for older imports."""
