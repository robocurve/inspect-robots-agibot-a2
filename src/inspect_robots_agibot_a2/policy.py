"""GO-1 HTTP and openpi websocket policy clients for the A2 contract."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_agibot_a2 import packing
from inspect_robots_agibot_a2.config import (
    Go1Config,
    OpenpiConfig,
    action_box,
    observation_space,
)

OPENPI_CLIENT_INSTALL_COMMAND = (
    'pip install "openpi-client @ '
    "git+https://github.com/Physical-Intelligence/openpi.git"
    '#subdirectory=packages/openpi-client"'
)

PostFn = Callable[[str, Mapping[str, Any]], np.ndarray]
OpenpiObservation = Mapping[str, Any]
OpenpiResponse = Mapping[str, Any]
InferFn = Callable[[OpenpiObservation], OpenpiResponse]


def _default_post(  # pragma: no cover - live GO-1 server transport
    url: str, payload: Mapping[str, Any]
) -> np.ndarray:
    import json_numpy
    import requests

    response = requests.post(
        url,
        data=json_numpy.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=120.0,
    )
    response.raise_for_status()
    return np.asarray(response.json())


def _default_infer(cfg: OpenpiConfig) -> InferFn:  # pragma: no cover - live websocket transport
    try:
        from openpi_client import image_tools, websocket_client_policy
    except ModuleNotFoundError as exc:
        if exc.name != "openpi_client" and not (exc.name or "").startswith("openpi_client."):
            raise
        raise ModuleNotFoundError(
            "The Physical Intelligence openpi-client is git-only. Install it with: "
            f"{OPENPI_CLIENT_INSTALL_COMMAND}",
            name=exc.name,
        ) from exc

    client = websocket_client_policy.WebsocketClientPolicy(
        host=cfg.host,
        port=cfg.port,
        api_key=cfg.api_key,
    )

    def infer(observation: OpenpiObservation) -> OpenpiResponse:
        payload = dict(observation)
        for key in (
            "observation/head_image",
            "observation/left_image",
            "observation/right_image",
        ):
            payload[key] = image_tools.resize_with_pad(
                np.asarray(payload[key]), cfg.resize_px, cfg.resize_px
            )
        response: OpenpiResponse = client.infer(payload)
        return response

    return infer


def _required_observation(
    observation: Observation, policy_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        head, left, right = (
            np.asarray(observation.images[name]) for name in ("head_cam", "left_cam", "right_cam")
        )
    except KeyError as exc:
        raise ValueError(f"observation missing camera {exc} required by {policy_name}") from exc
    if packing.STATE_KEY not in observation.state:
        raise ValueError(f"observation missing state key {packing.STATE_KEY!r}")
    state = np.asarray(observation.state[packing.STATE_KEY])
    packing.validate_dim(state)
    return head, left, right, state


def _validate_actions(actions: Any, source: str) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != packing.TOTAL_DIM:
        raise ValueError(
            f"{source} returned actions of shape {array.shape}; expected (N, {packing.TOTAL_DIM})"
        )
    if array.shape[0] == 0:
        raise ValueError(f"{source} returned an empty action chunk")
    if not np.isfinite(array).all():
        raise ValueError(f"{source} returned non-finite actions")
    return array


class Go1Policy:
    """Client for an AgiBot GO-1 ``POST /act`` policy server."""

    def __init__(
        self,
        config: Go1Config | None = None,
        *,
        post_fn: PostFn | None = None,
        clock: Callable[[], float] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else Go1Config.from_kwargs(**flat)
        self._post_fn: PostFn = post_fn if post_fn is not None else _default_post
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._instruction: str | None = None
        self.num_inferences = 0
        self.info = PolicyInfo(
            name=self._cfg.name,
            action_space=action_box(),
            observation_space=observation_space(),
            control_hz=None,
        )
        self.config = PolicyConfig(
            action_horizon=self._cfg.action_horizon,
            replan_interval=self._cfg.replan_interval,
        )

    def reset(self, scene: Scene) -> None:
        """Stash the instruction and reset inference accounting."""
        self._instruction = scene.instruction
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        """Send the exact GO-1 A2 payload and parse its bare-list response."""
        head, left, right, state = _required_observation(observation, self._cfg.name)
        payload: dict[str, Any] = {
            "top": head,
            "right": right,
            "left": left,
            "instruction": self._instruction or "",
            "state": state,
            "ctrl_freqs": np.asarray([self._cfg.ctrl_freqs]),
        }
        started = self._clock()
        response = self._post_fn(self._cfg.url, payload)
        elapsed = self._clock() - started
        actions = _validate_actions(response, "/act")[: self._cfg.action_horizon]
        self.num_inferences += 1
        return ActionChunk(
            actions=[Action(data=row.copy()) for row in actions],
            control_hz=float(self._cfg.ctrl_freqs),
            inference_latency_s=elapsed,
        )


class OpenpiPolicy:
    """Client for bring-your-own absolute-position A2 openpi fine-tunes."""

    def __init__(
        self,
        config: OpenpiConfig | None = None,
        *,
        infer_fn: InferFn | None = None,
        clock: Callable[[], float] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else OpenpiConfig.from_kwargs(**flat)
        self._infer_fn = infer_fn
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._instruction: str | None = None
        self.num_inferences = 0
        self.info = PolicyInfo(
            name=self._cfg.name,
            action_space=action_box(),
            observation_space=observation_space(),
            control_hz=None,
        )
        self.config = PolicyConfig(
            action_horizon=self._cfg.action_horizon,
            replan_interval=self._cfg.replan_interval,
        )

    def _infer(self) -> InferFn:
        if self._infer_fn is None:
            self._infer_fn = _default_infer(self._cfg)
        return self._infer_fn

    def reset(self, scene: Scene) -> None:
        """Stash the instruction and reset inference accounting."""
        self._instruction = scene.instruction
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        """Return pass-through absolute 16-D actions without polarity changes."""
        head, left, right, state = _required_observation(observation, self._cfg.name)
        request: dict[str, Any] = {
            "observation/head_image": head,
            "observation/left_image": left,
            "observation/right_image": right,
            "observation/joint_position": state,
            "prompt": self._instruction or "",
        }
        started = self._clock()
        response = self._infer()(request)
        elapsed = self._clock() - started
        if "actions" not in response:
            raise ValueError("openpi response missing 'actions'")
        actions = _validate_actions(response["actions"], "openpi")[: self._cfg.action_horizon]
        self.num_inferences += 1
        return ActionChunk(
            actions=[Action(data=row.copy()) for row in actions],
            control_hz=30.0,
            inference_latency_s=elapsed,
        )
