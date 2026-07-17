from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from inspect_robots_agibot_a2._ros import ROS_INSTALL_GUIDANCE, _load_rclpy, image_to_array


def _message(
    data: bytes,
    *,
    encoding: str = "rgb8",
    height: int = 1,
    width: int = 2,
    step: int = 6,
    big_endian: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        encoding=encoding,
        height=height,
        width=width,
        step=step,
        is_bigendian=big_endian,
    )


def test_load_rclpy_success(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = SimpleNamespace(ok=lambda: True)
    monkeypatch.setitem(sys.modules, "rclpy", sentinel)
    assert _load_rclpy() is sentinel


def test_load_rclpy_guidance_and_unrelated_missing_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def missing_rclpy(name: str, *args: object, **kwargs: object) -> object:
        if name == "rclpy":
            raise ModuleNotFoundError("missing", name="rclpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_rclpy)
    monkeypatch.delitem(sys.modules, "rclpy", raising=False)
    with pytest.raises(ModuleNotFoundError, match="ROS 2 Humble") as exc_info:
        _load_rclpy()
    assert "ROS_DOMAIN_ID=232" in str(exc_info.value)
    assert "FastRTPS" in ROS_INSTALL_GUIDANCE

    def missing_dependency(name: str, *args: object, **kwargs: object) -> object:
        if name == "rclpy":
            raise ModuleNotFoundError("dependency", name="some_dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="dependency"):
        _load_rclpy()


def test_rgb8_and_padded_rows_decode() -> None:
    raw = bytes([1, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12, 88, 88])
    image = image_to_array(_message(raw, height=2, width=2, step=8))
    assert image.dtype == np.uint8
    assert image.tolist() == [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]


def test_bgr8_converts_to_rgb() -> None:
    image = image_to_array(_message(bytes([1, 2, 3, 4, 5, 6]), encoding="BGR8"))
    assert image.tolist() == [[[3, 2, 1], [6, 5, 4]]]


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (_message(b"", encoding="mono8"), "unsupported image encoding"),
        (_message(b"", big_endian=True), "big-endian"),
        (_message(b"", height=0), "invalid"),
        (_message(b"", step=5), "invalid"),
        (_message(b"\x00" * 5), "shorter"),
    ],
)
def test_image_conversion_rejects_invalid_messages(message: SimpleNamespace, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        image_to_array(message)
