from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest
from inspect_robots.policy import PolicyConfig
from inspect_robots.scene import Scene
from inspect_robots.types import Observation

from inspect_robots_agibot_a2.config import Go1Config, OpenpiConfig
from inspect_robots_agibot_a2.policy import Go1Policy, OpenpiPolicy


def _observation(state: np.ndarray | None = None) -> Observation:
    vector = (
        np.asarray(
            [
                0.11,
                -0.22,
                0.33,
                -0.44,
                0.15,
                -0.16,
                0.17,
                0.81,
                -0.91,
                1.02,
                -1.13,
                1.24,
                -0.25,
                0.26,
                -0.17,
                0.38,
            ],
            dtype=np.float32,
        )
        if state is None
        else state
    )
    return Observation(
        images={
            "head_cam": np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
            "left_cam": np.full((2, 3, 3), 17, dtype=np.uint8),
            "right_cam": np.full((2, 3, 3), 29, dtype=np.uint8),
        },
        state={"joint_pos": vector},
    )


def test_both_policy_infos_and_configs_hide_secrets() -> None:
    go1 = Go1Policy(Go1Config(action_horizon=7, replan_interval=3))
    openpi = OpenpiPolicy(OpenpiConfig(api_key="secret"))
    assert go1.info.name == "go1"
    assert openpi.info.name == "openpi"
    assert go1.info.control_hz is openpi.info.control_hz is None
    assert go1.config == PolicyConfig(action_horizon=7, replan_interval=3)
    assert openpi.config == PolicyConfig(action_horizon=16, replan_interval=8)
    assert "api_key" not in asdict(openpi.config)


def test_go1_wire_payload_is_exact_and_state_is_verbatim() -> None:
    captured: dict[str, Any] = {}
    state = _observation().state["joint_pos"]

    def post(url: str, payload: dict[str, Any]) -> np.ndarray:
        captured["url"] = url
        captured["payload"] = payload
        return np.asarray([[float(index) for index in range(16)]])

    clock = iter([2.0, 2.375]).__next__
    policy = Go1Policy(post_fn=post, clock=clock)
    policy.reset(Scene(id="s", instruction="move the red cube"))
    chunk = policy.act(_observation())
    payload = captured["payload"]
    assert captured["url"] == "http://127.0.0.1:9000/act"
    assert list(payload) == ["top", "right", "left", "instruction", "state", "ctrl_freqs"]
    assert payload["instruction"] == "move the red cube"
    assert payload["state"].dtype == state.dtype
    assert payload["state"].tobytes() == state.tobytes()
    assert np.array_equal(payload["top"], _observation().images["head_cam"])
    assert np.array_equal(payload["left"], _observation().images["left_cam"])
    assert np.array_equal(payload["right"], _observation().images["right_cam"])
    assert np.array_equal(payload["ctrl_freqs"], np.asarray([30]))
    assert chunk.actions[0].data.tolist() == list(map(float, range(16)))
    assert chunk.control_hz == 30.0
    assert chunk.inference_latency_s == pytest.approx(0.375)
    assert policy.num_inferences == 1


def test_go1_parses_bare_list_truncates_and_threads_empty_instruction() -> None:
    captured: dict[str, Any] = {}
    actions = [[float(row)] * 16 for row in range(4)]

    def post(_url: str, payload: dict[str, Any]) -> np.ndarray:
        captured.update(payload)
        return np.asarray(actions)

    policy = Go1Policy(Go1Config(action_horizon=2), post_fn=post, clock=lambda: 0.0)
    chunk = policy.act(_observation())
    assert captured["instruction"] == ""
    assert len(chunk) == 2
    policy.reset(Scene(id="s", instruction="again"))
    assert policy.num_inferences == 0


def test_openpi_request_and_absolute_action_pass_through() -> None:
    captured: dict[str, Any] = {}
    actions = np.arange(48, dtype=np.float64).reshape(3, 16) / 17.0

    def infer(payload: dict[str, Any]) -> dict[str, np.ndarray]:
        captured.update(payload)
        return {"actions": actions}

    policy = OpenpiPolicy(
        OpenpiConfig(action_horizon=2), infer_fn=infer, clock=iter([1.0, 1.2]).__next__
    )
    policy.reset(Scene(id="s", instruction="lift"))
    chunk = policy.act(_observation())
    assert list(captured) == [
        "observation/head_image",
        "observation/left_image",
        "observation/right_image",
        "observation/joint_position",
        "prompt",
    ]
    assert captured["prompt"] == "lift"
    assert (
        captured["observation/joint_position"].tobytes()
        == _observation().state["joint_pos"].tobytes()
    )
    assert len(chunk) == 2
    assert np.array_equal(chunk.actions[0].data, actions[0])
    assert np.array_equal(chunk.actions[1].data, actions[1])
    assert chunk.control_hz == 30.0
    assert chunk.inference_latency_s == pytest.approx(0.2)
    assert policy.num_inferences == 1


def test_openpi_real_infer_factory_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[OpenpiConfig] = []

    def factory(cfg: OpenpiConfig):
        calls.append(cfg)
        return lambda _payload: {"actions": np.zeros((1, 16))}

    monkeypatch.setattr("inspect_robots_agibot_a2.policy._default_infer", factory)
    policy = OpenpiPolicy(clock=lambda: 0.0)
    assert calls == []
    policy.act(_observation())
    assert calls == [policy._cfg]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (np.zeros((2, 15)), r"expected \(N, 16\)"),
        (np.zeros(16), r"expected \(N, 16\)"),
        (np.zeros((0, 16)), "empty action chunk"),
        (np.full((1, 16), np.nan), "non-finite"),
    ],
)
@pytest.mark.parametrize("kind", ["go1", "openpi"])
def test_action_shape_empty_and_finite_validation(
    response: np.ndarray, message: str, kind: str
) -> None:
    if kind == "go1":
        policy = Go1Policy(post_fn=lambda _url, _payload: response, clock=lambda: 0.0)
    else:
        policy = OpenpiPolicy(infer_fn=lambda _payload: {"actions": response}, clock=lambda: 0.0)
    with pytest.raises(ValueError, match=message):
        policy.act(_observation())


def test_openpi_missing_actions() -> None:
    policy = OpenpiPolicy(infer_fn=lambda _payload: {}, clock=lambda: 0.0)
    with pytest.raises(ValueError, match="missing 'actions'"):
        policy.act(_observation())


@pytest.mark.parametrize("kind", ["go1", "openpi"])
def test_helpful_missing_observation_errors(kind: str) -> None:
    if kind == "go1":
        policy: Any = Go1Policy(post_fn=lambda _url, _payload: np.zeros((1, 16)), clock=lambda: 0.0)
    else:
        policy = OpenpiPolicy(
            infer_fn=lambda _payload: {"actions": np.zeros((1, 16))}, clock=lambda: 0.0
        )
    with pytest.raises(ValueError, match="missing camera"):
        policy.act(Observation(images={}, state={"joint_pos": np.zeros(16)}))
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="missing state key"):
        policy.act(
            Observation(
                images={"head_cam": image, "left_cam": image, "right_cam": image},
                state={},
            )
        )
    with pytest.raises(ValueError, match="expected a 16-D vector"):
        policy.act(
            Observation(
                images={"head_cam": image, "left_cam": image, "right_cam": image},
                state={"joint_pos": np.zeros(15)},
            )
        )
