"""Validated configs and shared spaces for the AgiBot A2 plugin."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace

from inspect_robots_agibot_a2.packing import DIM_LABELS, STATE_SPEC, TOTAL_DIM

_T = TypeVar("_T", bound="_FromKwargs")

# Transcribed from AimDK A2 v1.3 Motion Control on 2026-07-17:
# https://open.agibot.com/docs/en/aimdk/a2/v1_3/dev_guide/07-01-motion_control
# The documented arm rows are:
# L J1 [-2.91, 2.91], J2 [-0.46, 1.60], J3 [-2.91, 2.91],
# J4 [-2.00, -0.03], J5 [-2.94, 2.94], J6 [-0.45, 0.45],
# J7 [-0.35, 0.35]; R J1 [-2.91, 2.91], J2 [-1.60, 0.46],
# J3 [-2.91, 2.94] (asymmetric 2.94 transcribed verbatim), J4 [0.03, 2.00],
# J5 [-2.94, 2.94], J6 [-0.45, 0.45], J7 [0.35, 0.35]. The final J7 row is
# an evident documentation typo and should read [-0.35, 0.35]. Shipped defaults
# tighten both J6 rows to [-0.3, 0.3] and both J7 rows to [-0.2, 0.2], the same
# page's recommended parallel-wrist workspace. Gripper slots use [0, 1].
DEFAULT_JOINT_LOW: tuple[float, ...] = (
    -2.91,
    -0.46,
    -2.91,
    -2.00,
    -2.94,
    -0.3,
    -0.2,
    0.0,
    -2.91,
    -1.60,
    -2.91,
    0.03,
    -2.94,
    -0.3,
    -0.2,
    0.0,
)
DEFAULT_JOINT_HIGH: tuple[float, ...] = (
    2.91,
    1.60,
    2.91,
    -0.03,
    2.94,
    0.3,
    0.2,
    1.0,
    2.91,
    0.46,
    2.94,
    2.00,
    2.94,
    0.3,
    0.2,
    1.0,
)
DEFAULT_HOME_POSE: tuple[float, ...] = (
    0.0,
    0.0,
    0.0,
    -1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    1.0,
)

ACTION_SEMANTICS = ActionSemantics(
    control_mode="joint_pos",
    rotation_repr="none",
    gripper="continuous",
    frame="base",
    dim_labels=DIM_LABELS,
)


class _FromKwargs:
    """Build frozen dataclasses from flat CLI-friendly keyword arguments."""

    _FLOAT_TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def from_kwargs(cls: type[_T], **flat: Any) -> _T:
        """Reject unknown keys and parse configured comma-separated tuples."""
        names = {field.name for field in dataclasses.fields(cls)}  # type: ignore[arg-type]
        unknown = set(flat) - names
        if unknown:
            raise TypeError(f"{cls.__name__} got unexpected config keys: {sorted(unknown)}")
        for key in cls._FLOAT_TUPLE_FIELDS & set(flat):
            value = flat[key]
            if isinstance(value, str):
                try:
                    flat[key] = tuple(float(part) for part in value.split(","))
                except ValueError:
                    raise ValueError(
                        f"{key} must be a comma-separated list of numbers, got {value!r}"
                    ) from None
        return cls(**flat)


@dataclass(frozen=True)
class A2Config(_FromKwargs):
    """A2 topic, safety, streaming, hand, mode, and operator configuration."""

    _FLOAT_TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"joint_low", "joint_high", "home_pose", "rest_pose"}
    )

    mode_rpc_url: str = "http://192.168.100.100:56322"
    arm_topic: str = "/motion/control/arm_joint_command"
    hand_topic: str = "/motion/control/hand_joint_command"
    arm_state_topic: str = "/motion/control/arm_joint_state"
    hand_state_topic: str = "/motion/control/hand_joint_state"
    head_cam_topic: str = "/aima/hal/rgbd_camera/head_front/color"
    left_cam_topic: str = "/aima/hal/fish_eye_camera/chest_left/color"
    right_cam_topic: str = "/aima/hal/fish_eye_camera/chest_right/color"
    domain_id: int = 232
    control_hz: float = 30.0
    stream_hz: float = 100.0
    joint_low: tuple[float, ...] = DEFAULT_JOINT_LOW
    joint_high: tuple[float, ...] = DEFAULT_JOINT_HIGH
    home_pose: tuple[float, ...] = DEFAULT_HOME_POSE
    rest_pose: tuple[float, ...] | None = None
    max_joint_speed: float = 3.0
    thumb_swing_count: int = 0
    hand_torque: int = 2000
    hand_deadband: float = 0.05
    unattended: bool = False
    docs_extra: str = ""
    require_servo_mode: bool = True

    def __post_init__(self) -> None:
        """Reject values that violate the fixed 16-D transport contract."""
        for name in ("control_hz", "stream_hz", "max_joint_speed"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("joint_low", "joint_high", "home_pose"):
            if len(getattr(self, name)) != TOTAL_DIM:
                raise ValueError(f"{name} must have {TOTAL_DIM} entries")
        if self.rest_pose is not None and len(self.rest_pose) != TOTAL_DIM:
            raise ValueError(f"rest_pose must have {TOTAL_DIM} entries")
        low, high = self.low, self.high
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("joint_low and joint_high must contain only finite values")
        if np.any(low >= high):
            raise ValueError("joint_low must be below joint_high in every dimension")
        for name in ("home_pose", "rest_pose"):
            pose = getattr(self, name)
            if pose is not None:
                values = np.asarray(pose, dtype=np.float64)
                if not np.isfinite(values).all() or np.any(values < low) or np.any(values > high):
                    raise ValueError(f"{name} must be finite and inside joint_low/joint_high")
        micro_commands = math.ceil(self.stream_hz / self.control_hz)
        if 1.0 / (self.control_hz * micro_commands) > 0.03:
            raise ValueError("control_hz and stream_hz produce an inter-command gap above 0.03 s")
        if not isinstance(self.domain_id, int) or isinstance(self.domain_id, bool):
            raise ValueError("domain_id must be an integer")
        if not self.mode_rpc_url:
            raise ValueError("mode_rpc_url must not be empty")
        for name in (
            "arm_topic",
            "hand_topic",
            "arm_state_topic",
            "hand_state_topic",
            "head_cam_topic",
            "left_cam_topic",
            "right_cam_topic",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        for name, upper in (("thumb_swing_count", 2000), ("hand_torque", 5700)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= upper:
                raise ValueError(f"{name} must be an integer in [0, {upper}]")
        if not np.isfinite(self.hand_deadband) or not 0 <= self.hand_deadband < 1:
            raise ValueError("hand_deadband must be finite and in [0, 1)")

    @property
    def low(self) -> npt.NDArray[np.float64]:
        """Return configured lower bounds as float64."""
        return np.asarray(self.joint_low, dtype=np.float64)

    @property
    def high(self) -> npt.NDArray[np.float64]:
        """Return configured upper bounds as float64."""
        return np.asarray(self.joint_high, dtype=np.float64)

    @property
    def micro_commands(self) -> int:
        """Return the ceil-rule micro-command count per policy step."""
        return math.ceil(self.stream_hz / self.control_hz)


@dataclass(frozen=True)
class Go1Config(_FromKwargs):
    """GO-1 HTTP transport metadata for normative 16-D A2 fine-tunes."""

    server_url: str = "http://127.0.0.1:9000"
    endpoint: str = "/act"
    action_horizon: int = 30
    replan_interval: int | None = None
    name: str = "go1"
    ctrl_freqs: int = 30

    def __post_init__(self) -> None:
        """Reject malformed URLs and chunk metadata."""
        if not self.server_url or not self.endpoint or not self.name:
            raise ValueError("server_url, endpoint, and name must not be empty")
        if not self.endpoint.startswith("/"):
            raise ValueError("endpoint must start with '/'")
        for name in ("action_horizon", "ctrl_freqs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.replan_interval is not None and (
            not isinstance(self.replan_interval, int)
            or isinstance(self.replan_interval, bool)
            or self.replan_interval < 1
        ):
            raise ValueError("replan_interval must be None or a positive integer")

    @property
    def url(self) -> str:
        """Join the configured server base and fixed endpoint."""
        return self.server_url.rstrip("/") + "/" + self.endpoint.lstrip("/")


@dataclass(frozen=True)
class OpenpiConfig(_FromKwargs):
    """Openpi websocket metadata for absolute 16-D A2 fine-tunes."""

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    action_horizon: int = 16
    replan_interval: int = 8
    name: str = "openpi"
    resize_px: int = 224

    def __post_init__(self) -> None:
        """Reject invalid websocket and chunk metadata."""
        if not self.host or not self.name:
            raise ValueError("host and name must not be empty")
        if (
            not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("port must be an integer in [1, 65535]")
        for name in ("action_horizon", "replan_interval", "resize_px"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def action_box(cfg: A2Config | None = None) -> Box:
    """Build the shared absolute joint-position space."""
    return Box(
        shape=(TOTAL_DIM,),
        low=cfg.low if cfg is not None else None,
        high=cfg.high if cfg is not None else None,
        semantics=ACTION_SEMANTICS,
    )


def observation_space() -> ObservationSpace:
    """Build the three-camera packed-proprioception contract."""
    cameras = (
        CameraSpec(name="head_cam", height=720, width=1280, channels=3),
        CameraSpec(name="left_cam", height=480, width=640, channels=3),
        CameraSpec(name="right_cam", height=480, width=640, channels=3),
    )
    return ObservationSpace(cameras=cameras, state=STATE_SPEC)
