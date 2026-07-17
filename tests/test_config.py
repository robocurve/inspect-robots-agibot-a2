from __future__ import annotations

import math

import numpy as np
import pytest

from inspect_robots_agibot_a2.config import (
    ACTION_SEMANTICS,
    DEFAULT_HOME_POSE,
    DEFAULT_JOINT_HIGH,
    DEFAULT_JOINT_LOW,
    A2Config,
    Go1Config,
    OpenpiConfig,
    action_box,
    observation_space,
)
from inspect_robots_agibot_a2.packing import DIM_LABELS, STATE_KEY, TOTAL_DIM


def test_a2_pure_defaults_regression() -> None:
    cfg = A2Config()
    assert cfg.home_pose == DEFAULT_HOME_POSE
    assert cfg.rest_pose is None
    assert cfg.micro_commands == 4
    assert 1.0 / (cfg.control_hz * cfg.micro_commands) <= 0.03
    assert np.all(np.asarray(cfg.home_pose) >= cfg.low)
    assert np.all(np.asarray(cfg.home_pose) <= cfg.high)
    assert cfg.home_pose[3] == -1.0
    assert cfg.home_pose[11] == 1.0
    assert cfg.home_pose[7] == cfg.home_pose[15] == 1.0
    assert DEFAULT_JOINT_LOW[5:8] == (-0.3, -0.2, 0.0)
    assert DEFAULT_JOINT_HIGH[13:16] == (0.3, 0.2, 1.0)
    assert DEFAULT_JOINT_HIGH[10] == 2.94


def test_from_kwargs_rejects_unknown_and_parses_float_tuples() -> None:
    with pytest.raises(TypeError, match="unexpected config keys"):
        A2Config.from_kwargs(no_such_field=1)
    encoded = ",".join(str(value) for value in DEFAULT_HOME_POSE)
    cfg = A2Config.from_kwargs(home_pose=encoded, rest_pose=encoded)
    assert cfg.home_pose == pytest.approx(DEFAULT_HOME_POSE)
    assert cfg.rest_pose == pytest.approx(DEFAULT_HOME_POSE)
    with pytest.raises(ValueError, match="home_pose must be a comma-separated"):
        A2Config.from_kwargs(home_pose="0,bad")
    assert A2Config.from_kwargs(rest_pose=DEFAULT_HOME_POSE).rest_pose == DEFAULT_HOME_POSE


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"control_hz": 0.0}, "control_hz"),
        ({"stream_hz": np.inf}, "stream_hz"),
        ({"max_joint_speed": -1.0}, "max_joint_speed"),
        ({"joint_low": (0.0,) * 15}, "joint_low must have 16"),
        ({"joint_high": (0.0,) * 15}, "joint_high must have 16"),
        ({"home_pose": (0.0,) * 15}, "home_pose must have 16"),
        ({"rest_pose": (0.0,) * 15}, "rest_pose must have 16"),
        ({"joint_low": (*DEFAULT_JOINT_LOW[:-1], np.nan)}, "only finite"),
        ({"joint_low": DEFAULT_JOINT_HIGH}, "joint_low must be below"),
        ({"home_pose": (*DEFAULT_HOME_POSE[:-1], 2.0)}, "home_pose must be finite"),
        ({"rest_pose": (*DEFAULT_HOME_POSE[:-1], -1.0)}, "rest_pose must be finite"),
        ({"control_hz": 1.0, "stream_hz": 1.0}, "inter-command gap"),
        ({"domain_id": True}, "domain_id"),
        ({"mode_rpc_url": ""}, "mode_rpc_url"),
        ({"arm_topic": ""}, "arm_topic"),
        ({"thumb_swing_count": -1}, "thumb_swing_count"),
        ({"thumb_swing_count": True}, "thumb_swing_count"),
        ({"hand_torque": 5701}, "hand_torque"),
        ({"hand_deadband": -0.1}, "hand_deadband"),
        ({"hand_deadband": 1.0}, "hand_deadband"),
    ],
)
def test_a2_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        A2Config(**kwargs)


def test_ceil_rule_has_no_integer_multiple_constraint() -> None:
    cfg = A2Config(control_hz=29.0, stream_hz=97.0)
    assert cfg.micro_commands == math.ceil(97.0 / 29.0) == 4


def test_go1_defaults_url_and_from_kwargs_reject_url_property() -> None:
    cfg = Go1Config(server_url="http://host:9000/", endpoint="/act")
    assert cfg.url == "http://host:9000/act"
    assert (cfg.action_horizon, cfg.replan_interval, cfg.ctrl_freqs) == (30, None, 30)
    with pytest.raises(TypeError, match="unexpected config keys"):
        Go1Config.from_kwargs(url="http://bad")


@pytest.mark.parametrize(
    ("constructor", "kwargs", "message"),
    [
        (Go1Config, {"server_url": ""}, "must not be empty"),
        (Go1Config, {"endpoint": "act"}, "start with"),
        (Go1Config, {"action_horizon": 0}, "action_horizon"),
        (Go1Config, {"ctrl_freqs": True}, "ctrl_freqs"),
        (Go1Config, {"replan_interval": 0}, "replan_interval"),
        (OpenpiConfig, {"host": ""}, "must not be empty"),
        (OpenpiConfig, {"port": 0}, "port"),
        (OpenpiConfig, {"port": True}, "port"),
        (OpenpiConfig, {"action_horizon": 0}, "action_horizon"),
        (OpenpiConfig, {"replan_interval": True}, "replan_interval"),
        (OpenpiConfig, {"resize_px": 0}, "resize_px"),
    ],
)
def test_policy_config_validation(
    constructor: object, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        constructor(**kwargs)  # type: ignore[operator]


def test_shared_spaces_pin_semantics_state_cameras_and_bounds() -> None:
    cfg = A2Config()
    box = action_box(cfg)
    assert box.shape == (TOTAL_DIM,)
    assert np.array_equal(box.low, cfg.low)
    assert np.array_equal(box.high, cfg.high)
    assert box.semantics is ACTION_SEMANTICS
    assert ACTION_SEMANTICS.control_mode == "joint_pos"
    assert ACTION_SEMANTICS.rotation_repr == "none"
    assert ACTION_SEMANTICS.gripper == "continuous"
    assert ACTION_SEMANTICS.frame == "base"
    assert ACTION_SEMANTICS.dim_labels == DIM_LABELS
    assert action_box().low is action_box().high is None
    obs = observation_space()
    assert obs.camera_names == frozenset({"head_cam", "left_cam", "right_cam"})
    assert [(cam.height, cam.width) for cam in obs.cameras] == [
        (720, 1280),
        (480, 640),
        (480, 640),
    ]
    assert obs.state_keys == frozenset({STATE_KEY})
    assert obs.state is not None and obs.state.fields[0].shape == (16,)
