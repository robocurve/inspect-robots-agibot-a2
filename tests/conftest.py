from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from inspect_robots.scene import Scene


class FakeDriver:
    def __init__(self) -> None:
        self.arms = np.zeros(14, dtype=np.float64)
        self.hands = np.zeros(12, dtype=np.float64)
        self.arm_commands: list[np.ndarray] = []
        self.hand_commands: list[tuple[list[int], int]] = []
        self.disconnect_calls = 0
        self.disconnect_error: Exception | None = None

    def publish_arm(self, q14: np.ndarray) -> None:
        command = np.asarray(q14, dtype=np.float64).copy()
        self.arm_commands.append(command)
        self.arms = command

    def publish_hand(self, counts12: Sequence[int], torque: int) -> None:
        command = [int(value) for value in counts12]
        self.hand_commands.append((command, torque))
        self.hands = np.asarray(command, dtype=np.float64)

    def read_arm_joints(self) -> np.ndarray:
        return self.arms.copy()

    def read_hand_counts(self) -> np.ndarray:
        return self.hands.copy()

    def read_images(self) -> dict[str, np.ndarray]:
        return {
            "head_cam": np.full((2, 3, 3), 1, dtype=np.uint8),
            "left_cam": np.full((2, 3, 3), 2, dtype=np.uint8),
            "right_cam": np.full((2, 3, 3), 3, dtype=np.uint8),
        }

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        if self.disconnect_error is not None:
            raise self.disconnect_error


class FakeModeClient:
    def __init__(self, states: list[str] | None = None) -> None:
        self.states = states or ["PASSIVE_UPPER_BODY_JOINT_SERVO"]
        self.set_calls: list[str] = []
        self.get_calls = 0

    def set_action(self, state: str) -> None:
        self.set_calls.append(state)

    def get_action(self) -> str:
        index = min(self.get_calls, len(self.states) - 1)
        self.get_calls += 1
        return self.states[index]


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


@pytest.fixture
def scene() -> Scene:
    return Scene(id="scene", instruction="reach the cube")
