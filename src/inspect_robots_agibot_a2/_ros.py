"""Lazy ROS 2 loading and pure sensor image conversion."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

ROS_INSTALL_GUIDANCE = (
    "rclpy is provided by a system ROS 2 installation, not by this Python package. "
    "Install and source ROS 2 Humble, set ROS_DOMAIN_ID=232 and ROS_LOCALHOST_ONLY=0, "
    "then use the FastRTPS profile supplied on the A2 robot image."
)


def _load_rclpy() -> Any:
    """Import rclpy lazily and raise installation guidance when unavailable."""
    try:
        import rclpy
    except ModuleNotFoundError as exc:
        if exc.name != "rclpy" and not (exc.name or "").startswith("rclpy."):
            raise
        raise ModuleNotFoundError(ROS_INSTALL_GUIDANCE, name=exc.name) from exc
    return rclpy


def image_to_array(message: Any) -> npt.NDArray[np.uint8]:
    """Decode an uncompressed ROS Image carrying rgb8 or bgr8 pixels."""
    encoding = str(message.encoding).lower()
    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported image encoding {message.encoding!r}; expected rgb8 or bgr8")
    if bool(message.is_bigendian):
        raise ValueError("big-endian sensor_msgs/Image data is unsupported")
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    row_width = width * 3
    if height < 1 or width < 1 or step < row_width:
        raise ValueError("invalid sensor_msgs/Image dimensions or step")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size < height * step:
        raise ValueError("sensor_msgs/Image data is shorter than height * step")
    rows = raw[: height * step].reshape(height, step)
    image = rows[:, :row_width].reshape(height, width, 3).copy()
    if encoding == "bgr8":
        image = image[:, :, ::-1].copy()
    return cast(npt.NDArray[np.uint8], image)
