"""Video frame encoding — converts rendered frames to transmittable format."""

from __future__ import annotations

import base64
from typing import Literal

import cv2
import numpy as np


class FrameEncoder:
    """Encodes BGR frames to various formats for WebSocket transmission."""

    def __init__(
        self,
        format: Literal["raw", "jpeg", "png", "vp8"] = "jpeg",
        jpeg_quality: int = 85,
        output_size: tuple[int, int] = (512, 512),
    ):
        self.format = format
        self.jpeg_quality = jpeg_quality
        self.output_size = output_size  # (width, height)

    def encode(self, frame: np.ndarray) -> bytes:
        """Encode a BGR frame to the configured format.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Encoded bytes.
        """
        # Resize if needed
        h, w = frame.shape[:2]
        tw, th = self.output_size
        if w != tw or h != th:
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)

        if self.format == "raw":
            return frame.tobytes()
        elif self.format == "jpeg":
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            return buf.tobytes()
        elif self.format == "png":
            _, buf = cv2.imencode(".png", frame)
            return buf.tobytes()
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def encode_base64(self, frame: np.ndarray) -> str:
        """Encode frame and return as base64 string."""
        return base64.b64encode(self.encode(frame)).decode("ascii")
