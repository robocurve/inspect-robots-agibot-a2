"""Real AgiBot A2 dual-arm embodiment with fully injected seams."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from inspect_robots.conformance import DeviceSlot
from inspect_robots.embodiment import SELF_PACED, EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, Observation, StepResult

from inspect_robots_agibot_a2 import packing
from inspect_robots_agibot_a2._ros import ROS_INSTALL_GUIDANCE, _load_rclpy, image_to_array
from inspect_robots_agibot_a2.config import A2Config, action_box, observation_space
from inspect_robots_agibot_a2.operator import OperatorIO, default_poll_end

Vec = npt.NDArray[np.float64]
ImageMap = Mapping[str, npt.NDArray[np.uint8]]

SERVO_ACTION = "PASSIVE_UPPER_BODY_JOINT_SERVO"
ALLOWED_ACTIONS = frozenset(
    {
        "PASSIVE_UPPER_BODY_JOINT_SERVO",
        "RL_LOCOMOTION_ARM_EXT_JOINT_SERVO",
        "RL_WHOLE_BODY_EXT_JOINT_SERVO",
    }
)
MODE_POLL_ATTEMPTS = 100
MODE_POLL_INTERVAL_S = 0.05

_DOCS = """Two 7-DoF AgiBot A2 arms with one open-positive power-grasp scalar per
hand. Actions and state are absolute positions in the robot base frame. Arm
slots are radians and gripper slots are normalized from 0 closed to 1 open.
- left_j1: left arm joint 1.
- left_j2: left arm joint 2.
- left_j3: left arm joint 3.
- left_j4: left elbow joint 4.
- left_j5: left arm joint 5.
- left_j6: left parallel-wrist joint 6.
- left_j7: left parallel-wrist joint 7.
- left_gripper: mean power grasp of five left finger flexion joints.
- right_j1: right arm joint 1.
- right_j2: right arm joint 2.
- right_j3: right arm joint 3.
- right_j4: right elbow joint 4.
- right_j5: right arm joint 5.
- right_j6: right parallel-wrist joint 6.
- right_j7: right parallel-wrist joint 7.
- right_gripper: mean power grasp of five right finger flexion joints.
Keep people clear, use small changes, and re-check all observations after each
deliberate motion."""


@runtime_checkable
class TaskEnvelopeLike(Protocol):
    """Read-only task metadata accepted by the optional binding hook."""

    @property
    def max_steps(self) -> int:
        """Return the framework-enforced rollout horizon."""
        ...


@runtime_checkable
class Driver(Protocol):
    """Minimal A2 ROS topic surface used by the embodiment."""

    def publish_arm(self, q14: npt.NDArray[np.float64]) -> None:
        """Publish fourteen arm positions in fixed AimDK order."""
        ...

    def publish_hand(self, counts12: Sequence[int], torque: int) -> None:
        """Publish twelve hand positions and a shared conservative torque."""
        ...

    def read_arm_joints(self) -> npt.NDArray[np.float64]:
        """Read fourteen arm positions in fixed AimDK order."""
        ...

    def read_hand_counts(self) -> npt.NDArray[np.float64]:
        """Read twelve hand position counts in fixed AimDK order."""
        ...

    def read_images(self) -> dict[str, npt.NDArray[np.uint8]]:
        """Read the head and two chest cameras."""
        ...

    def disconnect(self) -> None:
        """Release ROS resources."""
        ...


@runtime_checkable
class ModeClient(Protocol):
    """A2 action-state RPC surface."""

    def set_action(self, state: str) -> None:
        """Request an asynchronous transition to ``state``."""
        ...

    def get_action(self) -> str:
        """Return the current action state."""
        ...


DriverFactory = Callable[[A2Config], Driver]


class _RequestsModeClient:
    """Thin AimDK v1.3 McActionService client."""

    def __init__(
        self,
        base_url: str,
        *,
        clock: Callable[[], float] = time.time,
        post_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._clock = clock
        self._post_fn = post_fn

    def _post(self, method: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._post_fn is None:  # pragma: no cover - live robot network
            import requests

            post_fn = requests.post
        else:
            post_fn = self._post_fn
        url = self._base_url + "/rpc/aimdk.protocol.McActionService/" + method
        response = post_fn(url, json=body, timeout=5.0)
        if response.status_code != 200:
            raise RuntimeError(f"AimDK {method} failed with HTTP {response.status_code}")
        decoded = response.json()
        if not isinstance(decoded, Mapping):
            raise RuntimeError(f"AimDK {method} returned a non-object response")
        return decoded

    def set_action(self, state: str) -> None:
        """Send the documented v1.3 SetAction request body."""
        if state not in ALLOWED_ACTIONS:
            raise ValueError(f"unsupported AimDK action state {state!r}")
        now = self._clock()
        seconds = int(now)
        nanos = int((now - seconds) * 1_000_000_000)
        body = {
            "header": {
                "timestamp": {
                    "seconds": seconds,
                    "nanos": nanos,
                    "ms_since_epoch": int(now * 1000),
                },
                "control_source": "ControlSource_SAFE",
            },
            "command": {
                "action": "McAction_USE_EXT_CMD",
                "ext_action": state,
            },
        }
        self._post("SetAction", body)

    def get_action(self) -> str:
        """Read the current action from the documented GetAction RPC."""
        decoded = self._post("GetAction", {})
        for container in (decoded, decoded.get("data", {})):
            if isinstance(container, Mapping):
                for key in ("ext_action", "action", "current_action"):
                    value = container.get(key)
                    if isinstance(value, str):
                        return value
        raise RuntimeError("AimDK GetAction response did not contain an action state")


def _default_mode_client(cfg: A2Config) -> ModeClient:  # pragma: no cover - live robot network
    return _RequestsModeClient(cfg.mode_rpc_url)


def _default_driver_factory(cfg: A2Config) -> Driver:  # pragma: no cover - real ROS hardware
    import os

    os.environ["ROS_DOMAIN_ID"] = str(cfg.domain_id)
    rclpy = _load_rclpy()
    from sensor_msgs.msg import Image, JointState

    if not rclpy.ok():
        rclpy.init(args=None)
    node = rclpy.create_node("inspect_robots_agibot_a2")
    arm_pub = node.create_publisher(JointState, cfg.arm_topic, 10)
    hand_pub = node.create_publisher(JointState, cfg.hand_topic, 10)
    arm_state: JointState | None = None
    hand_state: JointState | None = None
    images: dict[str, npt.NDArray[np.uint8]] = {}

    def arm_callback(message: JointState) -> None:
        nonlocal arm_state
        arm_state = message

    def hand_callback(message: JointState) -> None:
        nonlocal hand_state
        hand_state = message

    def image_callback(name: str) -> Callable[[Image], None]:
        def callback(message: Image) -> None:
            images[name] = image_to_array(message)

        return callback

    node.create_subscription(JointState, cfg.arm_state_topic, arm_callback, 10)
    node.create_subscription(JointState, cfg.hand_state_topic, hand_callback, 10)
    for name, topic in (
        ("head_cam", cfg.head_cam_topic),
        ("left_cam", cfg.left_cam_topic),
        ("right_cam", cfg.right_cam_topic),
    ):
        node.create_subscription(Image, topic, image_callback(name), 10)

    def spin_until(predicate: Callable[[], bool]) -> None:
        while not predicate():
            rclpy.spin_once(node, timeout_sec=0.1)

    def make_message(names: Sequence[str], positions: npt.ArrayLike) -> JointState:
        message = JointState()
        message.name = list(names)
        message.position = np.asarray(positions, dtype=np.float64).tolist()
        message.velocity = [0.0] * len(names)
        message.effort = [0.0] * len(names)
        return message

    class _RosDriver:
        def publish_arm(self, q14: npt.NDArray[np.float64]) -> None:
            arm_pub.publish(make_message(packing.ARM_JOINT_NAMES, q14))
            rclpy.spin_once(node, timeout_sec=0.0)

        def publish_hand(self, counts12: Sequence[int], torque: int) -> None:
            message = make_message(packing.HAND_JOINT_NAMES, counts12)
            message.effort = [float(torque)] * packing.HAND_WIDTH
            hand_pub.publish(message)
            rclpy.spin_once(node, timeout_sec=0.0)

        def read_arm_joints(self) -> npt.NDArray[np.float64]:
            spin_until(lambda: arm_state is not None)
            assert arm_state is not None
            return np.asarray(arm_state.position, dtype=np.float64)

        def read_hand_counts(self) -> npt.NDArray[np.float64]:
            spin_until(lambda: hand_state is not None)
            assert hand_state is not None
            return np.asarray(hand_state.position, dtype=np.float64)

        def read_images(self) -> dict[str, npt.NDArray[np.uint8]]:
            required = {"head_cam", "left_cam", "right_cam"}
            spin_until(lambda: required <= images.keys())
            return {name: images[name].copy() for name in required}

        def disconnect(self) -> None:
            node.destroy_node()

    return _RosDriver()


class A2Embodiment:
    """Inspect Robots embodiment for A2 dual-arm ROS topic control."""

    RUNTIME_REQUIREMENTS: ClassVar[Mapping[str, str]] = {
        "rclpy": ROS_INSTALL_GUIDANCE,
        "requests": "pip install requests",
    }
    DEVICE_SLOTS: ClassVar[tuple[DeviceSlot, ...]] = ()

    def __init__(
        self,
        config: A2Config | None = None,
        *,
        driver_factory: DriverFactory | None = None,
        mode_client: ModeClient | None = None,
        operator: OperatorIO | None = None,
        poll_end: Callable[[], bool] | None = None,
        clock: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else A2Config.from_kwargs(**flat)
        self._driver_factory: DriverFactory = driver_factory or _default_driver_factory
        self._mode_client = mode_client
        self._operator = operator if operator is not None else OperatorIO()
        self._poll_end: Callable[[], bool] = poll_end or default_poll_end
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._sleep: Callable[[float], None] = sleep_fn or time.sleep
        self._driver: Driver | None = None
        self._instruction: str | None = None
        self._last_published: Vec | None = None
        self._last_hand_commands: list[float | None] = [None, None]
        self._t_last = 0.0
        self._bound_max_steps: int | None = None
        self.num_steps = 0

        docs = _DOCS
        if self._cfg.docs_extra.strip():
            docs += "\n\n" + self._cfg.docs_extra.strip()
        self.info = EmbodimentInfo(
            name="a2_arms",
            action_space=action_box(self._cfg),
            observation_space=observation_space(),
            control_hz=self._cfg.control_hz,
            is_simulated=False,
            capabilities=frozenset({SELF_PACED}),
            docs=docs,
        )

    def bind_task(self, envelope: TaskEnvelopeLike) -> None:
        """Store the rollout horizon for operator-facing status."""
        self._bound_max_steps = int(envelope.max_steps)

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect lazily, enter servo mode, home safely, and observe."""
        del seed
        if self._driver is None:
            self._driver = self._driver_factory(self._cfg)
        if self._cfg.require_servo_mode:
            self._enter_servo_mode()
        self._seed_baseline()
        if not self._cfg.unattended:
            self._operator.wait_ready(
                "Arms will move to the home pose. Stand clear, then press Enter..."
            )
        home = packing.validate_dim(self._cfg.home_pose)
        self._ramp_arm_to(home)
        self._publish_hands(home, force=True)
        if not self._cfg.unattended:
            self._operator.wait_ready()
            horizon = self._horizon_seconds()
            limit = f" Max {horizon:.0f}s." if horizon is not None else ""
            self._operator.output_fn(
                "Running: press Enter to end the episode, then y/N to score." + limit
            )
        self._instruction = scene.instruction
        self.num_steps = 0
        return self._observe(scene.instruction)

    def step(self, action: Action) -> StepResult:
        """Clamp, stream delta-capped micro-commands, pace, and observe."""
        self._require_driver()
        baseline = self._require_baseline().copy()
        requested = packing.validate_dim(action.data)
        target = np.clip(requested, self._cfg.low, self._cfg.high)
        rate_limited = False
        for index in range(1, self._cfg.micro_commands + 1):
            desired = baseline + (target - baseline) * (index / self._cfg.micro_commands)
            rate_limited |= self._publish_arm_capped(desired)
        self._publish_hands(target)
        self.num_steps += 1
        observation = self._observe(self._instruction)
        info: dict[str, Any] = {}
        if rate_limited:
            info["rate_limited"] = True
        if not self._cfg.unattended and self._poll_end():
            success = self._operator.confirm_success()
            info["operator_confirmed"] = success
            return StepResult(
                observation=observation,
                terminated=True,
                termination_reason="success" if success else "failure",
                info=info,
            )
        return StepResult(observation=observation, terminated=False, info=info)

    def close(self) -> None:
        """Optionally park arm-only, then disconnect and clear all state."""
        self._bound_max_steps = None
        driver = self._driver
        if driver is None:
            return
        try:
            if self._cfg.rest_pose is not None:
                self._seed_baseline()
                rest = np.clip(
                    packing.validate_dim(self._cfg.rest_pose), self._cfg.low, self._cfg.high
                )
                self._ramp_arm_to(rest)
        finally:
            try:
                driver.disconnect()
            finally:
                self._driver = None
                self._last_published = None
                self._last_hand_commands = [None, None]

    def _enter_servo_mode(self) -> None:
        client = self._mode_client
        if client is None:
            client = _default_mode_client(self._cfg)
            self._mode_client = client
        client.set_action(SERVO_ACTION)
        for attempt in range(MODE_POLL_ATTEMPTS):
            if client.get_action() == SERVO_ACTION:
                return
            if attempt + 1 < MODE_POLL_ATTEMPTS:
                self._sleep(MODE_POLL_INTERVAL_S)
        raise RuntimeError(f"A2 did not enter {SERVO_ACTION} after {MODE_POLL_ATTEMPTS} polls")

    def _require_driver(self) -> Driver:
        if self._driver is None:
            raise RuntimeError("step() called before reset() (or after close())")
        return self._driver

    def _require_baseline(self) -> Vec:
        if self._last_published is None:
            raise RuntimeError("published-command baseline is unavailable")
        return self._last_published

    def _seed_baseline(self) -> None:
        driver = self._require_driver()
        arms = np.asarray(driver.read_arm_joints(), dtype=np.float64)
        if arms.ndim != 1 or arms.shape[0] != packing.ARM_DOF * 2:
            raise ValueError(f"driver returned arm joints of shape {arms.shape}; expected (14,)")
        hands = np.asarray(driver.read_hand_counts(), dtype=np.float64)
        if hands.ndim != 1 or hands.shape[0] != packing.HAND_WIDTH:
            raise ValueError(f"driver returned hand counts of shape {hands.shape}; expected (12,)")
        left_gripper, right_gripper = packing.observed_grippers(hands)
        left = np.concatenate((arms[: packing.ARM_DOF], [left_gripper]))
        right = np.concatenate((arms[packing.ARM_DOF :], [right_gripper]))
        self._last_published = packing.pack(left, right)
        self._last_hand_commands = [left_gripper, right_gripper]
        self._t_last = self._clock()

    def _publish_arm_capped(self, desired: npt.ArrayLike) -> bool:
        driver = self._require_driver()
        prior = self._require_baseline()
        target = packing.validate_dim(desired)
        command = target.copy()
        cap = self._cfg.max_joint_speed / (self._cfg.control_hz * self._cfg.micro_commands)
        revolute = np.ones(packing.TOTAL_DIM, dtype=bool)
        revolute[list(packing.GRIPPER_IDXS)] = False
        raw_delta = target[revolute] - prior[revolute]
        clipped_delta = np.clip(raw_delta, -cap, cap)
        limited = not np.array_equal(raw_delta, clipped_delta)
        command[revolute] = prior[revolute] + clipped_delta
        command[list(packing.GRIPPER_IDXS)] = target[list(packing.GRIPPER_IDXS)]
        driver.publish_arm(packing.arm_slots(command))
        self._last_published = command
        self._pace_micro()
        return limited

    def _ramp_arm_to(self, target: npt.ArrayLike) -> None:
        destination = packing.validate_dim(target)
        revolute = np.ones(packing.TOTAL_DIM, dtype=bool)
        revolute[list(packing.GRIPPER_IDXS)] = False
        while np.any(np.abs(destination[revolute] - self._require_baseline()[revolute]) > 0):
            self._publish_arm_capped(destination)

    def _publish_hands(self, target: npt.ArrayLike, *, force: bool = False) -> None:
        driver = self._require_driver()
        command = packing.validate_dim(target)
        grippers = [float(command[index]) for index in packing.GRIPPER_IDXS]
        changed = [
            force or prior is None or abs(value - prior) > self._cfg.hand_deadband
            for value, prior in zip(grippers, self._last_hand_commands, strict=True)
        ]
        if not any(changed):
            return
        sent = [
            value if is_changed or prior is None else prior
            for value, prior, is_changed in zip(
                grippers, self._last_hand_commands, changed, strict=True
            )
        ]
        driver.publish_hand(
            packing.hand_command(sent[0], sent[1], self._cfg.thumb_swing_count),
            self._cfg.hand_torque,
        )
        for index, is_changed in enumerate(changed):
            if is_changed:
                self._last_hand_commands[index] = grippers[index]

    def _pace_micro(self) -> None:
        interval = 1.0 / (self._cfg.control_hz * self._cfg.micro_commands)
        elapsed = self._clock() - self._t_last
        self._sleep(max(0.0, interval - elapsed))
        self._t_last = self._clock()

    def _horizon_seconds(self) -> float | None:
        if self._bound_max_steps is None:
            return None
        return self._bound_max_steps / self._cfg.control_hz

    def _observe(self, instruction: str | None) -> Observation:
        driver = self._require_driver()
        arms = np.asarray(driver.read_arm_joints(), dtype=np.float64)
        if arms.ndim != 1 or arms.shape[0] != packing.ARM_DOF * 2:
            raise ValueError(f"driver returned arm joints of shape {arms.shape}; expected (14,)")
        hands = np.asarray(driver.read_hand_counts(), dtype=np.float64)
        if hands.ndim != 1 or hands.shape[0] != packing.HAND_WIDTH:
            raise ValueError(f"driver returned hand counts of shape {hands.shape}; expected (12,)")
        left_gripper, right_gripper = packing.observed_grippers(hands)
        state = packing.pack(
            np.concatenate((arms[: packing.ARM_DOF], [left_gripper])),
            np.concatenate((arms[packing.ARM_DOF :], [right_gripper])),
        )
        images = {
            name: np.asarray(frame, dtype=np.uint8) for name, frame in driver.read_images().items()
        }
        observed_at = self._clock()
        return Observation(
            images=images,
            state={packing.STATE_KEY: state},
            instruction=instruction,
            image_times=dict.fromkeys(images, observed_at),
            state_time=observed_at,
        )
