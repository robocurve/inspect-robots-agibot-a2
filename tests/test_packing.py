from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_agibot_a2 import packing


def test_constants_and_labels_pin_the_blockwise_contract() -> None:
    assert (packing.ARM_DOF, packing.ARM_WIDTH, packing.TOTAL_DIM) == (7, 8, 16)
    assert (slice(0, 8), slice(8, 16)) == (packing.LEFT, packing.RIGHT)
    assert packing.GRIPPER_IDXS == (7, 15)
    assert packing.STATE_KEY == "joint_pos"
    assert len(packing.DIM_LABELS) == len(set(packing.DIM_LABELS)) == 16
    assert packing.DIM_LABELS[:2] == ("left_j1", "left_j2")
    assert packing.DIM_LABELS[-2:] == ("right_j7", "right_gripper")
    assert packing.STATE_SPEC.fields[0].shape == (16,)
    assert packing.STATE_SPEC.fields[0].unit == "rad+normalized"


def test_fixed_ros_joint_name_orders() -> None:
    assert packing.ARM_JOINT_NAMES[0] == "idx13_left_arm_joint1"
    assert packing.ARM_JOINT_NAMES[-1] == "idx26_right_arm_joint7"
    assert packing.HAND_JOINT_NAMES == (
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


@pytest.mark.parametrize("bad", [np.zeros(15), np.zeros(17), np.zeros((2, 8))])
def test_validate_dim_is_strict(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="expected a 16-D vector"):
        packing.validate_dim(bad)


def test_pack_split_and_arm_slots_are_left_then_right() -> None:
    left = np.arange(8, dtype=float)
    right = np.arange(10, 18, dtype=float)
    packed = packing.pack(left, right)
    assert np.array_equal(packed, np.concatenate((left, right)))
    left_out, right_out = packing.split(packed)
    assert np.array_equal(left_out, left)
    assert np.array_equal(right_out, right)
    assert not np.shares_memory(left_out, packed)
    assert np.array_equal(
        packing.arm_slots(packed),
        np.asarray([0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]),
    )
    with pytest.raises(ValueError, match="expected a 8-D vector"):
        packing.pack(np.zeros(7), right)


@pytest.mark.parametrize(
    ("wire", "counts"),
    [(-0.4, 2000), (0.0, 2000), (0.137, 1726), (0.65, 700), (1.0, 0), (1.4, 0)],
)
def test_gripper_to_counts_inverts_scales_and_clips(wire: float, counts: int) -> None:
    assert packing.gripper_to_counts(wire) == counts


@pytest.mark.parametrize(
    ("counts", "wire"),
    [(-50, 1.0), (0, 1.0), (246, 0.877), (1300, 0.35), (2000, 0.0), (2400, 0.0)],
)
def test_counts_to_gripper_inverts_scales_and_clips(counts: float, wire: float) -> None:
    assert packing.counts_to_gripper(counts) == pytest.approx(wire)


def test_hand_command_and_observation_exclude_thumb_swing() -> None:
    command = packing.hand_command(0.8, 0.25, 123)
    assert command == [123, 400, 400, 400, 400, 400, 123, 1500, 1500, 1500, 1500, 1500]
    observed = np.asarray([1999, 100, 200, 300, 400, 500, 1, 1100, 1200, 1300, 1400, 1500])
    left, right = packing.observed_grippers(observed)
    assert left == pytest.approx(0.85)
    assert right == pytest.approx(0.35)
