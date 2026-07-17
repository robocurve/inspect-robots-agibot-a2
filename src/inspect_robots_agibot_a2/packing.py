"""Canonical 16-D packing for the AgiBot A2 dual arms and hands.

Each eight-slot side is seven absolute revolute positions in radians followed
by one normalized power-grasp value. The full vector is left then right. A
gripper value of 1 means open. This order and polarity are the normative wire
contract for A2 GO-1 and openpi fine-tunes used with this package.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import StateField, StateSpec

ARM_DOF = 7
ARM_WIDTH = ARM_DOF + 1
TOTAL_DIM = ARM_WIDTH * 2
LEFT = slice(0, ARM_WIDTH)
RIGHT = slice(ARM_WIDTH, TOTAL_DIM)
GRIPPER_IDXS = (ARM_WIDTH - 1, TOTAL_DIM - 1)
STATE_KEY = "joint_pos"

DIM_LABELS: tuple[str, ...] = tuple(
    f"{side}_{part}"
    for side in ("left", "right")
    for part in (*(f"j{i}" for i in range(1, ARM_DOF + 1)), "gripper")
)

ARM_JOINT_NAMES: tuple[str, ...] = (
    "idx13_left_arm_joint1",
    "idx14_left_arm_joint2",
    "idx15_left_arm_joint3",
    "idx16_left_arm_joint4",
    "idx17_left_arm_joint5",
    "idx18_left_arm_joint6",
    "idx19_left_arm_joint7",
    "idx20_right_arm_joint1",
    "idx21_right_arm_joint2",
    "idx22_right_arm_joint3",
    "idx23_right_arm_joint4",
    "idx24_right_arm_joint5",
    "idx25_right_arm_joint6",
    "idx26_right_arm_joint7",
)

# AimDK's fixed ROS JointState wire order. Each thumb swing is followed by the
# five flexion joints collapsed into this package's one gripper scalar.
HAND_JOINT_NAMES: tuple[str, ...] = (
    "L_thumb_swing_joint",
    "L_thumb_1_joint",
    "L_index_1_joint",
    "L_middle_1_joint",
    "L_ring_1_joint",
    "L_little_1_joint",
    "R_thumb_swing_joint",
    "R_thumb_1_joint",
    "R_index_1_joint",
    "R_middle_1_joint",
    "R_ring_1_joint",
    "R_little_1_joint",
)
HAND_WIDTH = len(HAND_JOINT_NAMES)
LEFT_HAND = slice(0, 6)
RIGHT_HAND = slice(6, 12)

STATE_SPEC = StateSpec(
    fields=(StateField(key=STATE_KEY, shape=(TOTAL_DIM,), unit="rad+normalized"),)
)

Vec = npt.NDArray[np.float64]


def validate_dim(vec: npt.ArrayLike, n: int = TOTAL_DIM) -> Vec:
    """Return a strict one-dimensional float vector of length ``n``."""
    arr: Vec = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"expected a {n}-D vector, got shape {np.shape(vec)}")
    return arr


def pack(left: npt.ArrayLike, right: npt.ArrayLike) -> Vec:
    """Pack two eight-slot side vectors in left-then-right order."""
    return np.concatenate((validate_dim(left, ARM_WIDTH), validate_dim(right, ARM_WIDTH)))


def split(vec: npt.ArrayLike) -> tuple[Vec, Vec]:
    """Split a packed vector into independent left and right copies."""
    arr = validate_dim(vec)
    return arr[LEFT].copy(), arr[RIGHT].copy()


def arm_slots(vec: npt.ArrayLike) -> Vec:
    """Return the 14 revolute slots in the fixed AimDK topic order."""
    left, right = split(vec)
    return np.concatenate((left[:ARM_DOF], right[:ARM_DOF]))


def gripper_to_counts(wire: float) -> int:
    """Convert open-positive normalized gripper units to AimDK hand counts."""
    clipped = float(np.clip(wire, 0.0, 1.0))
    return round((1.0 - clipped) * 2000.0)


def counts_to_gripper(counts: float) -> float:
    """Convert AimDK hand counts to open-positive normalized gripper units."""
    clipped = float(np.clip(counts, 0.0, 2000.0))
    return 1.0 - clipped / 2000.0


def hand_command(left_gripper: float, right_gripper: float, thumb_swing_count: int) -> list[int]:
    """Build the fixed 12-slot command with each thumb swing held constant."""
    left = gripper_to_counts(left_gripper)
    right = gripper_to_counts(right_gripper)
    return [thumb_swing_count, *([left] * 5), thumb_swing_count, *([right] * 5)]


def observed_grippers(counts: npt.ArrayLike) -> tuple[float, float]:
    """Collapse five flexion counts per hand, excluding each thumb swing."""
    arr = validate_dim(counts, HAND_WIDTH)
    left = counts_to_gripper(float(np.mean(arr[LEFT_HAND][1:])))
    right = counts_to_gripper(float(np.mean(arr[RIGHT_HAND][1:])))
    return left, right
