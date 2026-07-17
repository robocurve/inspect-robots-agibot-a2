# inspect-robots-agibot-a2 agent guide

Inspect Robots adapters for real AgiBot A2 dual arms driven by GO-1 HTTP or
openpi websocket policy servers. The framework lives in
[inspect-robots](https://github.com/robocurve/inspect-robots).

## The central contract

This package ships three registered components:

- `go1` passes three cameras, instruction, and state to a GO-1 `/act` server.
- `openpi` passes the same observation to an A2-specific openpi fine-tune.
- `a2_arms` streams commands through documented AimDK ROS 2 topics.

All components share a 16-D absolute `joint_pos` vector. Each eight-slot arm
block has seven radians followed by one normalized gripper scalar with 1 open.
The left block always precedes the right block.

## Layout

- `src/inspect_robots_agibot_a2/` contains the package and local module map.
- `tests/` contains fully injected hardware-free tests.
- `plans/0001-a2-go1-design.md` is the accepted binding design.

## Working here

- Set `UV_CACHE_DIR=$PWD/.uv-cache` for every uv command in this workspace.
- Install with `uv sync --locked --extra dev`.
- Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`,
  and `uv run pytest --cov` before handing off.
- Keep strict mypy and 100% statement and branch coverage.
- Keep ROS, requests, json_numpy, and openpi imports lazy so minimal package
  import needs only Inspect Robots and NumPy.

## Safety invariants

- `A2Embodiment.step()` clamps every target to configured absolute limits.
- Each revolute micro-command is independently rate-capped relative to the
  last command actually published.
- Reset and close-time parking reseed the baseline from a fresh arm reading.
- Gripper slots are excluded from the rad/s cap and gated per hand.
- Construction performs no hardware, network, camera, or stdin work.
- Success reaches scoring only as `termination_reason="success"`.
- The embodiment declares `SELF_PACED` and paces every micro-command.
- Both policy declarations keep `control_hz=None` to avoid compatibility
  warnings. Returned chunks carry execution-rate metadata.

## CI and releases

- CI installs from `uv.lock`. Run `uv lock` after dependency changes.
- `ci-ok` must need quality, test, import-hygiene, and openpi-seam.
- The openpi client is installed only from the Physical Intelligence Git URL.
- Versions come from git tags through hatch-vcs. Do not add a static project
  version to `pyproject.toml`.

## Writing style

- Do not use em dashes in prose. Use periods, commas, colons, or parentheses.
- Use bold only for definition-list leads and critical safety instructions.
- Do not use decorative emoji, slogans, chiasmus, or the prohibited contrast
  construction described in the repository requirements.
- Headers have no trailing colons.
