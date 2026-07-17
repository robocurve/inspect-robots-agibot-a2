from __future__ import annotations

import re

import inspect_robots_agibot_a2

EXPECTED_API = {
    "A2Config",
    "Go1Config",
    "OpenpiConfig",
    "A2Embodiment",
    "Go1Policy",
    "OpenpiPolicy",
    "OperatorIO",
    "STATE_KEY",
    "TOTAL_DIM",
    "DIM_LABELS",
    "build",
    "run_preflight",
    "__version__",
}


def test_public_api_is_exact_and_importable() -> None:
    assert set(inspect_robots_agibot_a2.__all__) == EXPECTED_API
    for name in inspect_robots_agibot_a2.__all__:
        assert hasattr(inspect_robots_agibot_a2, name)


def test_version_is_tag_derived_shape() -> None:
    assert re.match(r"\d+\.\d+", inspect_robots_agibot_a2.__version__)


def test_all_entry_points_resolve() -> None:
    from inspect_robots.registry import resolve

    assert resolve("policy", "go1").info.name == "go1"
    assert resolve("policy", "openpi").info.name == "openpi"
    assert resolve("embodiment", "a2_arms").info.name == "a2_arms"
