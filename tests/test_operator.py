from __future__ import annotations

import pytest
from inspect_robots.errors import EmbodimentFault

from inspect_robots_agibot_a2.operator import OperatorIO, default_poll_end


def test_wait_ready_reads_injected_prompt() -> None:
    prompts: list[str] = []
    OperatorIO(input_fn=lambda prompt: prompts.append(prompt) or "").wait_ready("ready?")
    assert prompts == ["ready?"]


@pytest.mark.parametrize("exception", [EOFError, OSError])
def test_wait_ready_hardens_dead_stdin(exception: type[Exception]) -> None:
    def fail(_prompt: str) -> str:
        raise exception("closed")

    with pytest.raises(EmbodimentFault, match=r"A2Config\(unattended=True\)"):
        OperatorIO(input_fn=fail).wait_ready()


@pytest.mark.parametrize("answer", ["y", "Yes", "1", "TRUE", "success", "pass"])
def test_confirm_success_affirmative(answer: str) -> None:
    assert OperatorIO(input_fn=lambda _prompt: answer).confirm_success() is True


@pytest.mark.parametrize("answer", ["n", "no", "", "nope"])
def test_confirm_success_negative(answer: str) -> None:
    assert OperatorIO(input_fn=lambda _prompt: answer).confirm_success() is False


@pytest.mark.parametrize("exception", [EOFError, OSError])
def test_confirm_success_hardens_dead_stdin(exception: type[Exception]) -> None:
    def fail(_prompt: str) -> str:
        raise exception("closed")

    with pytest.raises(EmbodimentFault, match="success prompt"):
        OperatorIO(input_fn=fail).confirm_success()


def test_default_poll_end_is_exposed() -> None:
    assert callable(default_poll_end)
