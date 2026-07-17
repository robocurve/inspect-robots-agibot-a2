from __future__ import annotations

import re

from inspect_robots_agibot_a2.config import A2Config
from inspect_robots_agibot_a2.embodiment import A2Embodiment
from inspect_robots_agibot_a2.packing import DIM_LABELS


def test_docs_name_every_dimension_exactly_once_without_numeric_bounds() -> None:
    docs = A2Embodiment().info.docs
    assert docs is not None
    for label in DIM_LABELS:
        assert docs.count(f"- {label}:") == 1
    assert re.search(r"\[[+-]?\d", docs) is None
    assert "normalized from 0 closed to 1 open" in docs


def test_docs_extra_is_stripped_and_appended() -> None:
    base = A2Embodiment().info.docs
    extended = A2Embodiment(A2Config(docs_extra="  Rig note.  ")).info.docs
    assert extended == base + "\n\nRig note."
    assert A2Embodiment(A2Config(docs_extra="  ")).info.docs == base
