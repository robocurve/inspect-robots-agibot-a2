# `inspect_robots_agibot_a2` package module map

The package supplies the `a2_arms` embodiment, `go1` and `openpi` policies, and
the shared 16-D absolute joint-position contract.

## Modules

| Module | Responsibility |
|---|---|
| `packing.py` | Pure constants, strict validation, blockwise packing, hand conversion, and fixed ROS wire order. |
| `config.py` | Frozen configs, AimDK limit defaults, validation, and shared space builders. |
| `embodiment.py` | Lazy ROS driver, action-state gate, clamps, streaming, hand gating, observations, and operator verdicts. |
| `policy.py` | Exact GO-1 HTTP client and lazy openpi websocket client. |
| `_ros.py` | Guided rclpy loader and pure rgb8 or bgr8 image decoding. |
| `operator.py` | Injectable readiness and success prompts with dead-stdin handling. |
| `preflight.py` | Hardware-free compatibility CLI. |
| `__init__.py` | Reviewed public API fenced by `__all__`. |

## Invariants

- Construction performs no hardware or network I/O.
- `step()` enforces absolute and per-micro-command rate limits without relying
  on a framework approver.
- Rate limiting always uses the last published command as its baseline.
- GO-1 state passes through without reordering or polarity conversion.
- Gripper counts convert only at the driver boundary, with 1 open publicly.
- Only `termination_reason="success"` reports success to a scorer.

## Writing style

- Do not use em dashes in prose.
- Do not use decorative emoji or canned contrast constructions.
- Headers have no trailing colons.
