from __future__ import annotations

import numpy as np
from inspect_robots import eval as robots_eval

from conftest import FakeDriver, FakeModeClient
from inspect_robots_agibot_a2.config import A2Config
from inspect_robots_agibot_a2.embodiment import A2Embodiment
from inspect_robots_agibot_a2.operator import OperatorIO
from inspect_robots_agibot_a2.policy import Go1Policy


def test_full_eval_propagates_success_with_fake_driver_and_post_fn() -> None:
    driver = FakeDriver()
    policy = Go1Policy(
        post_fn=lambda _url, _payload: np.asarray([A2Config().home_pose]),
        clock=lambda: 0.0,
    )
    embodiment = A2Embodiment(
        A2Config(max_joint_speed=300.0),
        driver_factory=lambda _cfg: driver,
        mode_client=FakeModeClient(),
        operator=OperatorIO(input_fn=lambda _prompt: "yes", output_fn=lambda _message: None),
        poll_end=lambda: True,
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
    )
    logs = robots_eval("cubepick-reach", policy, embodiment, sinks=[], seed=0)
    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    assert log.results.metrics["success_at_end"] == 1.0
    assert log.eval.policy_config == {
        "action_horizon": 30,
        "replan_interval": None,
        "temperature": None,
    }
    assert embodiment.num_steps == 1
