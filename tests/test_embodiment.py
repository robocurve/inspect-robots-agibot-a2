from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inspect_robots.conformance import missing_runtime_requirements
from inspect_robots.embodiment import SELF_PACED
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from conftest import FakeDriver, FakeModeClient, FakeTime
from inspect_robots_agibot_a2 import packing
from inspect_robots_agibot_a2.config import A2Config
from inspect_robots_agibot_a2.embodiment import (
    MODE_POLL_ATTEMPTS,
    MODE_POLL_INTERVAL_S,
    SERVO_ACTION,
    A2Embodiment,
    _RequestsModeClient,
)
from inspect_robots_agibot_a2.operator import OperatorIO


def _fast_config(**kwargs: Any) -> A2Config:
    return A2Config(max_joint_speed=300.0, unattended=True, **kwargs)


def _reset(
    driver: FakeDriver,
    scene: Scene,
    *,
    cfg: A2Config | None = None,
    mode: FakeModeClient | None = None,
    fake_time: FakeTime | None = None,
    **kwargs: Any,
) -> A2Embodiment:
    timer = fake_time or FakeTime()
    embodiment = A2Embodiment(
        cfg or _fast_config(),
        driver_factory=lambda _cfg: driver,
        mode_client=mode or FakeModeClient(),
        clock=timer.clock,
        sleep_fn=timer.sleep,
        **kwargs,
    )
    embodiment.reset(scene)
    driver.arm_commands.clear()
    driver.hand_commands.clear()
    timer.sleeps.clear()
    return embodiment


def _home() -> np.ndarray:
    return np.asarray(A2Config().home_pose, dtype=np.float64)


def test_init_is_inert_and_declares_self_pacing() -> None:
    calls: list[A2Config] = []
    embodiment = A2Embodiment(driver_factory=lambda cfg: calls.append(cfg))  # type: ignore[arg-type,func-returns-value]
    assert calls == []
    assert embodiment.info.name == "a2_arms"
    assert embodiment.info.capabilities == frozenset({SELF_PACED})
    assert embodiment.info.control_hz == 30.0


def test_reset_connects_lazily_polls_mode_and_reseeds_each_episode(scene: Scene) -> None:
    driver = FakeDriver()
    mode = FakeModeClient(["PASSIVE", "PASSIVE", SERVO_ACTION])
    timer = FakeTime()
    embodiment = A2Embodiment(
        _fast_config(),
        driver_factory=lambda _cfg: driver,
        mode_client=mode,
        clock=timer.clock,
        sleep_fn=timer.sleep,
    )
    assert driver.arm_commands == []
    observation = embodiment.reset(scene)
    assert mode.set_calls == [SERVO_ACTION]
    assert mode.get_calls == 3
    assert timer.sleeps[:2] == [MODE_POLL_INTERVAL_S, MODE_POLL_INTERVAL_S]
    assert observation.instruction == "reach the cube"
    driver.arms[0] = 0.4
    mode.states = [SERVO_ACTION]
    embodiment.reset(scene)
    assert mode.set_calls == [SERVO_ACTION, SERVO_ACTION]
    assert driver.arm_commands[-1][0] == pytest.approx(0.0)


def test_mode_gate_can_be_skipped(scene: Scene) -> None:
    driver = FakeDriver()
    mode = FakeModeClient()
    _reset(driver, scene, cfg=_fast_config(require_servo_mode=False), mode=mode)
    assert mode.set_calls == []
    assert mode.get_calls == 0


def test_mode_poll_is_bounded(scene: Scene) -> None:
    timer = FakeTime()
    embodiment = A2Embodiment(
        _fast_config(),
        driver_factory=lambda _cfg: FakeDriver(),
        mode_client=FakeModeClient(["PASSIVE"]),
        clock=timer.clock,
        sleep_fn=timer.sleep,
    )
    with pytest.raises(RuntimeError, match=f"after {MODE_POLL_ATTEMPTS} polls"):
        embodiment.reset(scene)
    assert len(timer.sleeps) == MODE_POLL_ATTEMPTS - 1


class _Response:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self.payload


def test_requests_mode_client_uses_documented_body_and_rpc_join() -> None:
    calls: list[tuple[str, dict[str, Any], float]] = []
    responses = iter([_Response({}), _Response({"data": {"action": SERVO_ACTION}})])

    def post(url: str, *, json: dict[str, Any], timeout: float) -> _Response:
        calls.append((url, json, timeout))
        return next(responses)

    client = _RequestsModeClient("http://robot:56322/", clock=lambda: 12.25, post_fn=post)
    client.set_action(SERVO_ACTION)
    assert client.get_action() == SERVO_ACTION
    set_url, body, timeout = calls[0]
    assert set_url.endswith("/rpc/aimdk.protocol.McActionService/SetAction")
    assert timeout == 5.0
    assert body == {
        "header": {
            "timestamp": {"seconds": 12, "nanos": 250000000, "ms_since_epoch": 12250},
            "control_source": "ControlSource_SAFE",
        },
        "command": {
            "action": "McAction_USE_EXT_CMD",
            "ext_action": SERVO_ACTION,
        },
    }
    assert calls[1][0].endswith("/rpc/aimdk.protocol.McActionService/GetAction")
    assert calls[1][1] == {}


def test_requests_mode_client_validation_and_response_errors() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        _RequestsModeClient(
            "http://robot", post_fn=lambda *_args, **_kwargs: _Response({})
        ).set_action("BAD")
    with pytest.raises(RuntimeError, match="HTTP 503"):
        _RequestsModeClient(
            "http://robot", post_fn=lambda *_args, **_kwargs: _Response({}, 503)
        ).get_action()
    with pytest.raises(RuntimeError, match="non-object"):
        _RequestsModeClient(
            "http://robot", post_fn=lambda *_args, **_kwargs: _Response([])
        ).get_action()
    root = _RequestsModeClient(
        "http://robot", post_fn=lambda *_args, **_kwargs: _Response({"ext_action": "ROOT"})
    )
    assert root.get_action() == "ROOT"
    with pytest.raises(RuntimeError, match="did not contain"):
        _RequestsModeClient(
            "http://robot", post_fn=lambda *_args, **_kwargs: _Response({"data": []})
        ).get_action()


def test_embodiment_lazily_builds_default_mode_client(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode = FakeModeClient()
    monkeypatch.setattr(
        "inspect_robots_agibot_a2.embodiment._default_mode_client", lambda _cfg: mode
    )
    driver = FakeDriver()
    embodiment = A2Embodiment(
        _fast_config(),
        driver_factory=lambda _cfg: driver,
        clock=lambda: 0.0,
        sleep_fn=lambda _delay: None,
    )
    embodiment.reset(scene)
    assert mode.set_calls == [SERVO_ACTION]


def test_default_homing_ramp_count_and_rate_cap(scene: Scene) -> None:
    driver = FakeDriver()
    timer = FakeTime()
    embodiment = A2Embodiment(
        A2Config(unattended=True),
        driver_factory=lambda _cfg: driver,
        mode_client=FakeModeClient(),
        clock=timer.clock,
        sleep_fn=timer.sleep,
    )
    embodiment.reset(scene)
    assert len(driver.arm_commands) == 40
    deltas = np.diff(np.vstack((np.zeros(14), *driver.arm_commands)), axis=0)
    assert np.max(np.abs(deltas)) <= 3.0 / 120.0 + 1e-12
    assert np.array_equal(driver.arms, packing.arm_slots(_home()))
    assert len(timer.sleeps) == 40


def test_interpolation_uses_ceil_rule_and_hand_computed_targets(scene: Scene) -> None:
    driver = FakeDriver()
    cfg = _fast_config(control_hz=30.0, stream_hz=100.0)
    embodiment = _reset(driver, scene, cfg=cfg)
    baseline = _home()
    target = baseline.copy()
    target[0] = 0.4
    target[8] = -0.8
    result = embodiment.step(Action(data=target))
    assert result.info == {}
    assert len(driver.arm_commands) == 4
    for index, command in enumerate(driver.arm_commands, start=1):
        expected = baseline + (target - baseline) * index / 4
        assert command == pytest.approx(packing.arm_slots(expected))


def test_per_micro_rate_cap_and_cross_step_last_published_continuity(scene: Scene) -> None:
    driver = FakeDriver()
    cfg = A2Config(unattended=True)
    embodiment = _reset(driver, scene, cfg=cfg)
    first = _home()
    first[0] = 1.0
    result1 = embodiment.step(Action(data=first))
    assert result1.info["rate_limited"] is True
    assert [command[0] for command in driver.arm_commands] == pytest.approx(
        [0.025, 0.05, 0.075, 0.1]
    )
    driver.arm_commands.clear()
    second = _home()
    second[0] = -1.0
    result2 = embodiment.step(Action(data=second))
    assert result2.info["rate_limited"] is True
    assert driver.arm_commands[0][0] == pytest.approx(0.075)


def test_step_hard_clamps_bounds_without_approver(scene: Scene) -> None:
    driver = FakeDriver()
    embodiment = _reset(driver, scene, cfg=_fast_config())
    target = np.full(16, -999.0)
    embodiment.step(Action(data=target))
    assert driver.arms == pytest.approx(packing.arm_slots(A2Config().low))
    counts, _torque = driver.hand_commands[-1]
    assert counts[1:6] == [2000] * 5
    assert counts[7:] == [2000] * 5


def test_grippers_are_excluded_from_rad_per_second_cap(scene: Scene) -> None:
    driver = FakeDriver()
    cfg = A2Config(unattended=True, max_joint_speed=0.0001)
    driver.arms = packing.arm_slots(np.asarray(cfg.home_pose, dtype=np.float64))
    embodiment = _reset(driver, scene, cfg=cfg)
    target = _home()
    target[7] = 0.123
    target[15] = 0.789
    result = embodiment.step(Action(data=target))
    assert "rate_limited" not in result.info
    counts, torque = driver.hand_commands[-1]
    assert counts == [0, 1754, 1754, 1754, 1754, 1754, 0, 422, 422, 422, 422, 422]
    assert torque == 2000


def test_hand_deadband_is_per_hand_and_preserves_other_last_sent_value(scene: Scene) -> None:
    driver = FakeDriver()
    embodiment = _reset(driver, scene, cfg=_fast_config(hand_deadband=0.05))
    action = _home()
    action[7] = 0.90
    embodiment.step(Action(data=action))
    assert driver.hand_commands[-1][0] == [0, 200, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0]
    count = len(driver.hand_commands)
    action[15] = 0.97
    embodiment.step(Action(data=action))
    assert len(driver.hand_commands) == count
    action[15] = 0.80
    embodiment.step(Action(data=action))
    assert driver.hand_commands[-1][0] == [
        0,
        200,
        200,
        200,
        200,
        200,
        0,
        400,
        400,
        400,
        400,
        400,
    ]


def test_thumb_swing_wire_order_and_observed_flexion_mean(scene: Scene) -> None:
    driver = FakeDriver()
    driver.hands = np.asarray([1999, 100, 200, 300, 400, 500, 1, 1100, 1200, 1300, 1400, 1500])
    cfg = _fast_config(thumb_swing_count=321)
    embodiment = _reset(driver, scene, cfg=cfg)
    driver.hands = np.asarray([1999, 100, 200, 300, 400, 500, 1, 1100, 1200, 1300, 1400, 1500])
    observation = embodiment._observe("inspect")
    state = observation.state[packing.STATE_KEY]
    assert state[7] == pytest.approx(0.85)
    assert state[15] == pytest.approx(0.35)
    target = _home()
    target[7], target[15] = 0.5, 0.25
    embodiment.step(Action(data=target))
    assert driver.hand_commands[-1][0] == [
        321,
        1000,
        1000,
        1000,
        1000,
        1000,
        321,
        1500,
        1500,
        1500,
        1500,
        1500,
    ]


def test_micro_commands_are_paced_with_injected_clock(scene: Scene) -> None:
    driver = FakeDriver()
    timer = FakeTime()
    embodiment = _reset(driver, scene, fake_time=timer)
    embodiment.step(Action(data=_home()))
    assert timer.sleeps == pytest.approx([1.0 / 120.0] * 4)


@pytest.mark.parametrize(("answer", "reason"), [("yes", "success"), ("no", "failure")])
def test_operator_success_and_failure(scene: Scene, answer: str, reason: str) -> None:
    driver = FakeDriver()
    answers = iter(["", "", answer])
    output: list[str] = []
    operator = OperatorIO(input_fn=lambda _prompt: next(answers), output_fn=output.append)
    embodiment = A2Embodiment(
        A2Config(max_joint_speed=300.0),
        driver_factory=lambda _cfg: driver,
        mode_client=FakeModeClient(),
        operator=operator,
        poll_end=lambda: True,
        clock=lambda: 0.0,
        sleep_fn=lambda _delay: None,
    )
    embodiment.bind_task(SimpleNamespace(max_steps=60))
    embodiment.reset(scene)
    result = embodiment.step(Action(data=_home()))
    assert result.terminated is True
    assert result.termination_reason == reason
    assert result.info["operator_confirmed"] is (answer == "yes")
    assert "Max 2s" in output[-1]


def test_unattended_skips_operator_poll_and_horizon_without_binding(scene: Scene) -> None:
    driver = FakeDriver()
    polls: list[bool] = []
    embodiment = _reset(
        driver,
        scene,
        poll_end=lambda: polls.append(True) or True,
    )
    assert embodiment._horizon_seconds() is None
    result = embodiment.step(Action(data=_home()))
    assert result.terminated is False
    assert polls == []


def test_camera_passthrough_and_honest_observation(scene: Scene) -> None:
    driver = FakeDriver()
    embodiment = _reset(driver, scene)
    driver.arms = np.arange(14, dtype=float) / 10
    driver.hands = np.asarray([0, 200, 300, 400, 500, 600, 0, 1200, 1300, 1400, 1500, 1600])
    observation = embodiment._observe("new instruction")
    assert set(observation.images) == {"head_cam", "left_cam", "right_cam"}
    assert observation.images["right_cam"].dtype == np.uint8
    assert observation.instruction == "new instruction"
    assert observation.state[packing.STATE_KEY][:7] == pytest.approx(driver.arms[:7])
    assert observation.state[packing.STATE_KEY][8:15] == pytest.approx(driver.arms[7:])
    assert observation.image_times.keys() == observation.images.keys()


def test_close_parks_arm_only_reseeds_and_is_idempotent(scene: Scene) -> None:
    rest = list(A2Config().home_pose)
    rest[0] = 0.2
    driver = FakeDriver()
    cfg = _fast_config(rest_pose=tuple(rest), thumb_swing_count=456)
    embodiment = _reset(driver, scene, cfg=cfg)
    driver.arms[0] = -0.3
    hands_before = len(driver.hand_commands)
    embodiment.close()
    assert driver.arm_commands[0][0] == pytest.approx(0.2)
    assert len(driver.hand_commands) == hands_before
    assert driver.disconnect_calls == 1
    embodiment.close()
    assert driver.disconnect_calls == 1
    with pytest.raises(RuntimeError, match="before reset"):
        embodiment.step(Action(data=_home()))


def test_close_disconnects_and_clears_handle_when_park_or_disconnect_fails(scene: Scene) -> None:
    class FailingDriver(FakeDriver):
        fail_publish = False

        def publish_arm(self, q14: np.ndarray) -> None:
            if self.fail_publish:
                raise RuntimeError("park failed")
            super().publish_arm(q14)

    rest = A2Config().home_pose
    driver = FailingDriver()
    embodiment = _reset(driver, scene, cfg=_fast_config(rest_pose=rest))
    driver.arms[0] = 0.2
    driver.fail_publish = True
    with pytest.raises(RuntimeError, match="park failed"):
        embodiment.close()
    assert driver.disconnect_calls == 1
    assert embodiment._driver is None

    driver2 = FakeDriver()
    embodiment2 = _reset(driver2, scene)
    driver2.disconnect_error = RuntimeError("disconnect failed")
    with pytest.raises(RuntimeError, match="disconnect failed"):
        embodiment2.close()
    assert embodiment2._driver is None


def test_driver_shape_errors_and_internal_guards(scene: Scene) -> None:
    driver = FakeDriver()
    driver.arms = np.zeros(13)
    embodiment = A2Embodiment(
        _fast_config(require_servo_mode=False),
        driver_factory=lambda _cfg: driver,
        clock=lambda: 0.0,
        sleep_fn=lambda _delay: None,
    )
    with pytest.raises(ValueError, match=r"expected \(14,\)"):
        embodiment.reset(scene)
    driver.arms = np.zeros(14)
    driver.hands = np.zeros(11)
    with pytest.raises(ValueError, match=r"expected \(12,\)"):
        embodiment.reset(scene)
    with pytest.raises(RuntimeError, match="baseline"):
        embodiment._require_baseline()


def test_observe_shape_errors_after_reset(scene: Scene) -> None:
    driver = FakeDriver()
    embodiment = _reset(driver, scene)
    driver.arms = np.zeros(13)
    with pytest.raises(ValueError, match=r"expected \(14,\)"):
        embodiment._observe(None)
    driver.arms = np.zeros(14)
    driver.hands = np.zeros(11)
    with pytest.raises(ValueError, match=r"expected \(12,\)"):
        embodiment._observe(None)


def test_runtime_requirements_are_mapping_and_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import Mapping

    assert isinstance(A2Embodiment.RUNTIME_REQUIREMENTS, Mapping)
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    assert missing_runtime_requirements(A2Embodiment) == A2Embodiment.RUNTIME_REQUIREMENTS
    assert A2Embodiment.DEVICE_SLOTS == ()


def test_step_rejects_non_finite_actions(scene: Scene) -> None:
    driver = FakeDriver()
    embodiment = _reset(driver, scene)
    published_before = len(driver.arm_commands)
    bad = _home()
    bad[3] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        embodiment.step(Action(data=bad))
    assert len(driver.arm_commands) == published_before


def test_homing_ramp_raises_when_cap_rounds_to_zero(scene: Scene) -> None:
    driver = FakeDriver()
    home = _home()
    home[0] = 1.5
    cfg = A2Config(unattended=True, max_joint_speed=1e-15, home_pose=tuple(home))
    driver.arms = packing.arm_slots(_home())
    driver.arms[0] = 1.0
    with pytest.raises(RuntimeError, match="no progress"):
        _reset(driver, scene, cfg=cfg)


def test_reset_seeds_baseline_after_operator_wait(scene: Scene) -> None:
    driver = FakeDriver()
    moved = packing.arm_slots(_home())
    moved[0] = 0.4

    prompts: list[str] = []

    def input_fn(prompt: str = "") -> str:
        prompts.append(prompt)
        if len(prompts) == 1:
            driver.arms = moved.copy()
        return ""

    timer = FakeTime()
    embodiment = A2Embodiment(
        A2Config(unattended=False),
        driver_factory=lambda _cfg: driver,
        mode_client=FakeModeClient(),
        operator=OperatorIO(input_fn=input_fn, output_fn=lambda _msg: None),
        poll_end=lambda: False,
        clock=timer.clock,
        sleep_fn=timer.sleep,
    )
    embodiment.reset(scene)
    # The ramp must start from the pose observed AFTER the stand-clear wait
    # (0.4 was set during the prompt), so the first published command departs
    # from 0.4 toward home, never from the stale pre-wait zero.
    first = driver.arm_commands[0][0]
    assert 0.0 < abs(first - 0.4) < 0.4
    assert first != pytest.approx(0.0)
