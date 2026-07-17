"""EOF-hardened operator confirmation for real A2 runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from inspect_robots.errors import EmbodimentFault

_AFFIRMATIVE = frozenset({"y", "yes", "1", "true", "success", "pass"})


@dataclass
class OperatorIO:
    """Console I/O with injectable functions for hardware-free tests."""

    input_fn: Callable[[str], str] = input
    output_fn: Callable[[str], None] = print

    def wait_ready(self, prompt: str = "Position the scene, then press Enter to start...") -> None:
        """Wait for readiness or convert dead stdin into an embodiment fault."""
        try:
            self.input_fn(prompt)
        except (EOFError, OSError) as exc:
            raise EmbodimentFault(
                "operator readiness prompt could not read stdin (no interactive terminal?). "
                "Run from a real TTY, inject OperatorIO with a working input_fn, or set "
                "A2Config(unattended=True) (CLI: -E unattended=true)."
            ) from exc
        _drain_stdin()

    def confirm_success(self, prompt: str = "Did the robot succeed? [y/N]: ") -> bool:
        """Return whether the operator supplied an affirmative verdict."""
        try:
            answer = self.input_fn(prompt)
        except (EOFError, OSError) as exc:
            raise EmbodimentFault(
                "operator success prompt could not read stdin (no interactive terminal?)"
            ) from exc
        return answer.strip().lower() in _AFFIRMATIVE


def _drain_stdin() -> None:
    """Discard buffered terminal input so reset does not end on its first step."""
    import sys

    if not sys.stdin.isatty():
        return
    import select  # pragma: no cover - TTY-bound

    while select.select([sys.stdin], [], [], 0)[0]:  # pragma: no cover - TTY-bound
        sys.stdin.readline()  # pragma: no cover - TTY-bound


def default_poll_end() -> bool:  # pragma: no cover - requires a real TTY
    """Return whether the operator pressed Enter without blocking."""
    import select
    import sys

    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return False
    sys.stdin.readline()
    return True
