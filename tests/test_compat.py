from __future__ import annotations

from typing import Any

from inspect_robots.compat import check_compatibility
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import resolve
from inspect_robots.spaces import ActionSemantics, Box

from inspect_robots_agibot_a2.config import action_box, observation_space
from inspect_robots_agibot_a2.embodiment import A2Embodiment
from inspect_robots_agibot_a2.policy import Go1Policy, OpenpiPolicy


class _Policy:
    config = PolicyConfig()

    def __init__(self, info: PolicyInfo) -> None:
        self.info = info

    def reset(self, scene: object) -> None:
        return None

    def act(self, observation: object) -> Any:
        raise AssertionError("not called")


def test_both_pairs_have_zero_errors_and_zero_warnings() -> None:
    for policy in (Go1Policy(), OpenpiPolicy()):
        report = check_compatibility(policy, A2Embodiment())
        assert report.ok is True
        assert report.errors == []
        assert report.warnings == []


def test_builtin_cubepick_reach_is_realizable() -> None:
    task = resolve("task", "cubepick-reach")
    for policy in (Go1Policy(), OpenpiPolicy()):
        assert check_compatibility(policy, A2Embodiment(), task).errors == []


def test_wrong_dimension_is_a_hard_error() -> None:
    info = PolicyInfo(
        name="wrong",
        action_space=Box(shape=(15,), semantics=ActionSemantics(control_mode="joint_pos")),
    )
    report = check_compatibility(_Policy(info), A2Embodiment())  # type: ignore[arg-type]
    assert any(issue.code == "action_dim" for issue in report.errors)


def test_advertised_policy_rate_warns() -> None:
    info = PolicyInfo(
        name="rated",
        action_space=action_box(),
        observation_space=observation_space(),
        control_hz=25.0,
    )
    report = check_compatibility(_Policy(info), A2Embodiment())  # type: ignore[arg-type]
    assert report.ok is True
    assert [issue.code for issue in report.warnings] == ["control_rate"]
