"""Inspect Robots adapters for AgiBot A2 arms, GO-1, and openpi.

The package registers embodiment ``a2_arms`` and policies ``go1`` and
``openpi``. All three share one absolute 16-D joint-position contract and
remain inert at construction.
"""

from __future__ import annotations

from inspect_robots_agibot_a2.config import A2Config, Go1Config, OpenpiConfig
from inspect_robots_agibot_a2.embodiment import A2Embodiment
from inspect_robots_agibot_a2.operator import OperatorIO
from inspect_robots_agibot_a2.packing import DIM_LABELS, STATE_KEY, TOTAL_DIM
from inspect_robots_agibot_a2.policy import Go1Policy, OpenpiPolicy
from inspect_robots_agibot_a2.preflight import build, run_preflight

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("inspect-robots-agibot-a2")
except PackageNotFoundError:  # pragma: no cover - non-installed source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "DIM_LABELS",
    "STATE_KEY",
    "TOTAL_DIM",
    "A2Config",
    "A2Embodiment",
    "Go1Config",
    "Go1Policy",
    "OpenpiConfig",
    "OpenpiPolicy",
    "OperatorIO",
    "__version__",
    "build",
    "run_preflight",
]
