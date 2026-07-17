# Inspect Robots AgiBot A2

[![CI](https://github.com/robocurve/inspect-robots-agibot-a2/actions/workflows/ci.yml/badge.svg)](https://github.com/robocurve/inspect-robots-agibot-a2/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/inspect-robots-agibot-a2)](https://pypi.org/project/inspect-robots-agibot-a2/)
[![Python](https://img.shields.io/pypi/pyversions/inspect-robots-agibot-a2)](https://pypi.org/project/inspect-robots-agibot-a2/)

Inspect Robots adapters for real AgiBot A2 Ultra dual arms. The package
registers three components:

- `a2_arms`, an embodiment streaming absolute arm targets through documented
  AimDK ROS 2 topics.
- `go1`, an HTTP client for GO-1 A2 fine-tunes served through `POST /act`.
- `openpi`, a websocket client for bring-your-own A2 openpi fine-tunes.

All three use the same 16-D absolute `joint_pos` contract: seven left arm
joints, one left gripper scalar, seven right arm joints, and one right gripper
scalar. Revolute slots are radians. Gripper slots are normalized with 1 open.

This repository is a sibling of
[inspect-robots-franka](https://github.com/robocurve/inspect-robots-franka),
[inspect-robots-yam](https://github.com/robocurve/inspect-robots-yam), and the
[Inspect Robots framework](https://github.com/robocurve/inspect-robots).

## Install

### Client machine

Python 3.10 or newer is required.

```bash
pip install inspect-robots-agibot-a2
```

The package includes `requests` and `json-numpy` for the GO-1 transport. The
openpi client is distributed from its upstream Git repository:

```bash
pip install "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"
```

### Robot-side ROS 2 setup

The real driver uses the A2 robot's system ROS 2 Humble installation. `rclpy`
is intentionally absent from Python dependencies because it must match the
system ROS installation. Source ROS and configure discovery before running an
evaluation:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=232
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/the/profile/from/the/robot/image.xml
```

Disable the built-in motion player before sending joint commands. This RPC is
on port 56444, which differs from the action-state service on port 56322:

```bash
curl -X POST \
  http://192.168.100.100:56444/rpc/aimdk.protocol.MotionCommandService/DisableMotionPlayer \
  -H 'Content-Type: application/json' \
  -d '{}'
```

The embodiment requests `PASSIVE_UPPER_BODY_JOINT_SERVO` through the v1.3
`McActionService`, then polls `GetAction` until the transition completes. The
documented SetAction request is:

```json
{
  "header": {
    "timestamp": {
      "seconds": 0,
      "nanos": 0,
      "ms_since_epoch": 0
    },
    "control_source": "ControlSource_SAFE"
  },
  "command": {
    "action": "McAction_USE_EXT_CMD",
    "ext_action": "PASSIVE_UPPER_BODY_JOINT_SERVO"
  }
}
```

AimDK also documents `RL_LOCOMOTION_ARM_EXT_JOINT_SERVO` and
`RL_WHOLE_BODY_EXT_JOINT_SERVO` as compatible upper-body action states. This
package defaults to the passive upper-body state. Set `require_servo_mode=false`
only when an operator or supervisor establishes the mode out of band.

> [!WARNING]
> These RPCs and bodies are pinned to the AimDK A2 v1.3 documentation. AgiBot's
> public documentation has changed through v2.1. Verify the service ports,
> method names, action body, ROS topics, and limits against the software running
> on your robot before enabling motion.

### GPU machine

Run the AgiBot-World GO-1 deployment server on the GPU machine. The exact
checkpoint arguments depend on your A2 fine-tune. A typical invocation is:

```bash
python evaluate/deploy.py --port 9000
```

The client posts to `/act`. The server must accept json_numpy-encoded ndarrays
with keys `top`, `right`, `left`, `instruction`, `state`, and `ctrl_freqs`, and
return a bare JSON list shaped `(N, 16)`.

This package ships only the client. GO-1 weights are licensed CC BY-NC-SA and
are intended for research use. Review and follow the model license before
downloading, fine-tuning, serving, or publishing derived weights.

No public A2 openpi checkpoint is available. The `openpi` entry point is for an
A2 fine-tune that emits absolute 16-D actions in this package's packing order.

## Preflight

Preflight checks declared action dimensions, semantics, cameras, state, and an
optional task without connecting to the robot or policy server:

```bash
inspect-robots-agibot-a2-preflight --task cubepick-reach --dry-run
inspect-robots-agibot-a2-preflight --task cubepick-reach --dry-run --json
```

The JSON result always includes `dry_run`.

## Run on hardware

Create a framework configuration with policy and embodiment arguments. Every
ROS topic can be remapped for A2 variants without changing code:

```ini
[policy]
name = go1

[policy.args]
server_url = http://192.168.1.20:9000
endpoint = /act
ctrl_freqs = 30

[embodiment]
name = a2_arms

[embodiment.args]
mode_rpc_url = http://192.168.100.100:56322
arm_topic = /motion/control/arm_joint_command
hand_topic = /motion/control/hand_joint_command
arm_state_topic = /motion/control/arm_joint_state
hand_state_topic = /motion/control/hand_joint_state
head_cam_topic = /aima/hal/rgbd_camera/head_front/color
left_cam_topic = /aima/hal/fish_eye_camera/chest_left/color
right_cam_topic = /aima/hal/fish_eye_camera/chest_right/color
domain_id = 232
unattended = false
```

Then start an evaluation using your normal Inspect Robots CLI or Python
workflow. For Python:

```python
from inspect_robots import eval
from inspect_robots_agibot_a2 import A2Embodiment, Go1Policy

logs = eval(
    "cubepick-reach",
    Go1Policy(server_url="http://192.168.1.20:9000"),
    A2Embodiment(),
)
```

Construction is inert. The ROS driver and mode client connect during the first
`reset()`.

## Safety

> [!WARNING]
> This software commands a full-size humanoid robot. Keep an operator at the
> emergency stop, clear the full arm workspace, use conservative first motions,
> and validate every observation and command on your exact software version.

The embodiment enforces two independent safety bounds inside `step()`, even if
no framework approver is installed:

1. Every target is hard-clamped to `joint_low` and `joint_high`.
2. Every revolute micro-command is capped to
   `max_joint_speed / (control_hz * n)` relative to the last published command.

Here `n = ceil(stream_hz / control_hz)`. Defaults use 30 Hz policy steps and
100 Hz streaming, producing four paced micro-commands per step at about 8.3 ms
intervals. This satisfies the documented 0.03 s maximum gap while preserving a
30 Hz public action contract. The next step always begins from the last command
actually published, including after rate saturation. `info["rate_limited"]` is
true whenever any micro-command reaches the rate cap.

AimDK documents a 3 rad/s speed maximum, a 6.28 rad/s² acceleration maximum,
and recommends basic low-pass filtering. This version enforces the documented
speed maximum and uses interpolation to bound slew. It deliberately omits a
separate first-order low-pass filter. Validate acceleration and tracking on the
physical robot before increasing motion amplitude.

The gripper slots use normalized units and are excluded from the rad/s cap.
Each hand is deadband-gated independently. A single scalar deliberately
collapses five flexion joints into a power grasp, while the thumb-swing joint is
held at `thumb_swing_count`. Observations average the five flexion counts and
exclude thumb swing. This is a v1 simplification, not dexterous-hand control.
At the driver boundary, AimDK counts range from 0 open to 2000 closed, so the
public open-positive scalar is inverted and scaled in both directions.

The A2 Ultra camera map has one head camera and two chest fisheye cameras. It
has no wrist cameras. GO-1 checkpoints trained with wrist views are not
compatible without retraining or a deliberate camera adaptation.

Policy inference pauses the command stream between calls. A large GO-1 model
may take hundreds of milliseconds. Interpolation cannot fill that pause. On the
first hardware run, verify that the controller holds position during inference
and resumes tracking smoothly. The first resumed micro-command is rate-capped
against the last published command. If the controller faults on command gaps
instead of holding position, stop using this version. That behavior requires a
background keep-alive design in a future release.

First-run checklist:

- Verify the v1.3 RPCs, ports, action state, topic names, and joint limits
  against the robot's installed software.
- Disable the motion player and confirm the requested servo state through
  `GetAction`.
- Staff the emergency stop and clear the full-body workspace.
- Confirm all 14 observed arm angles and both open-positive gripper values.
- Confirm head, left chest, and right chest camera identity and orientation.
- Send a small arm-only jog, then a small per-hand gripper jog.
- Measure command gaps and confirm hold behavior during policy inference.
- Keep `unattended=false` until the whole setup has been validated.

## Configuration

### A2 embodiment fields

| Field | Default | Meaning |
|---|---:|---|
| `mode_rpc_url` | `http://192.168.100.100:56322` | Base URL for `McActionService` |
| `domain_id` | `232` | Required ROS domain ID |
| `control_hz` | `30.0` | Public policy-step rate |
| `stream_hz` | `100.0` | Requested micro-command stream rate |
| `max_joint_speed` | `3.0` | Revolute cap in rad/s for steps, home, and park |
| `thumb_swing_count` | `0` | Fixed swing count for both thumbs, range 0 to 2000 |
| `hand_torque` | `2000` | Shared conservative hand torque, maximum 5700 |
| `hand_deadband` | `0.05` | Per-hand normalized republish threshold |
| `home_pose` | elbow-safe open-hand pose | Reset target inside all shipped limits |
| `rest_pose` | `None` | Optional close-time arm-only park pose |
| `require_servo_mode` | `true` | Set and poll the passive upper-body servo state |
| `unattended` | `false` | Skip readiness and success prompts |
| `docs_extra` | empty | Markdown appended to agent-facing embodiment docs |

The arm command, hand command, arm state, hand state, and three camera topic
fields use the documented A2 Ultra names shown in the configuration example.

### GO-1 policy fields

| Field | Default | Meaning |
|---|---:|---|
| `server_url` | `http://127.0.0.1:9000` | GO-1 server base URL |
| `endpoint` | `/act` | Fixed action endpoint |
| `action_horizon` | `30` | Checkpoint chunk-length metadata |
| `replan_interval` | `None` | Execute full chunks by framework default |
| `ctrl_freqs` | `30` | One-element ndarray sent to GO-1 |
| `name` | `go1` | Evaluation log policy name |

### openpi policy fields

| Field | Default | Meaning |
|---|---:|---|
| `host` | `127.0.0.1` | Websocket server host |
| `port` | `8000` | Websocket server port |
| `api_key` | `None` | Optional authentication secret |
| `action_horizon` | `16` | Fine-tune chunk-length metadata |
| `replan_interval` | `8` | Framework replanning interval |
| `resize_px` | `224` | Resize-with-pad size in the default transport |
| `name` | `openpi` | Evaluation log policy name |

`api_key` stays in the private transport config and never enters framework
`PolicyConfig` or evaluation logs.

### Joint vector and shipped limits

The table below shows shipped defaults. The source limits were transcribed from
the AimDK A2 v1.3 motion-control page. Both J6 ranges are tightened to
plus or minus 0.3 and both J7 ranges to plus or minus 0.2 for the documented
parallel-wrist workspace. Verify these values against the robot's software
version because the public documentation has changed from v1.3 through v2.1.

| Slot | Label | Unit | Shipped low | Shipped high |
|---:|---|---|---:|---:|
| 0 | `left_j1` | rad | -2.91 | 2.91 |
| 1 | `left_j2` | rad | -0.46 | 1.60 |
| 2 | `left_j3` | rad | -2.91 | 2.91 |
| 3 | `left_j4` | rad | -2.00 | -0.03 |
| 4 | `left_j5` | rad | -2.94 | 2.94 |
| 5 | `left_j6` | rad | -0.30 | 0.30 |
| 6 | `left_j7` | rad | -0.20 | 0.20 |
| 7 | `left_gripper` | normalized, 1 open | 0.00 | 1.00 |
| 8 | `right_j1` | rad | -2.91 | 2.91 |
| 9 | `right_j2` | rad | -1.60 | 0.46 |
| 10 | `right_j3` | rad | -2.91 | 2.94 |
| 11 | `right_j4` | rad | 0.03 | 2.00 |
| 12 | `right_j5` | rad | -2.94 | 2.94 |
| 13 | `right_j6` | rad | -0.30 | 0.30 |
| 14 | `right_j7` | rad | -0.20 | 0.20 |
| 15 | `right_gripper` | normalized, 1 open | 0.00 | 1.00 |

The GO-1 state and action vectors pass through this order and polarity
verbatim. A compatible A2 fine-tune must use `state_dim=action_dim=16` and this
exact convention. Version 1 provides no reorder or polarity hooks.

## Development

Use a repository-local UV cache in restricted workspaces:

```bash
export UV_CACHE_DIR=$PWD/.uv-cache
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov
```

Tests use injected drivers, mode clients, transports, clocks, sleeps, operator
I/O, and end-of-episode polling. They perform no ROS, network, hardware, or
stdin operations. Coverage requires 100% statement and branch coverage.

## Citation

See [`CITATION.cff`](CITATION.cff), or cite the software repository and the
version used in your evaluation.

## License

This package is licensed under the MIT License. See [`LICENSE`](LICENSE).
Model weights and robot software retain their own licenses and terms.
