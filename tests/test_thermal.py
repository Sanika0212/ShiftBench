"""The throttle that keeps a scoring run from restarting the machine it runs on."""

from __future__ import annotations

import os
import time

import pytest

from shiftbench.thermal import DutyCycle, background_qos, limit_threads, memory_headroom


def test_limit_threads_caps_every_pool() -> None:
    limit_threads(2)
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_duty_cycle_idles_in_proportion_to_the_work() -> None:
    cycle = DutyCycle(duty=0.5)
    started = time.monotonic()
    for _ in range(3):
        with cycle:
            time.sleep(0.05)
    assert time.monotonic() - started > 0.25  # 0.15 s of work plus about as much idling
    assert 0.4 < cycle.share < 0.6


def test_a_full_duty_cycle_never_idles() -> None:
    cycle = DutyCycle(duty=1.0)
    started = time.monotonic()
    with cycle:
        time.sleep(0.05)
    assert time.monotonic() - started < 0.12
    assert cycle.paused == 0.0


def test_a_long_chunk_does_not_idle_forever() -> None:
    cycle = DutyCycle(duty=0.1, max_pause=0.05)
    with cycle:
        time.sleep(0.02)
    assert cycle.paused <= 0.05  # capped, not the 0.18 s the ratio would ask for


@pytest.mark.parametrize("duty", [0.0, -0.5, 1.5])
def test_an_impossible_duty_is_refused(duty: float) -> None:
    with pytest.raises(ValueError, match="duty"):
        DutyCycle(duty)


def test_background_qos_reports_whether_it_took() -> None:
    assert isinstance(background_qos(), bool)


def test_memory_headroom_is_a_fraction() -> None:
    assert 0.0 <= memory_headroom() <= 1.0


def test_memory_headroom_reports_full_when_the_platform_will_not_say(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guard built on this must never block a run on a machine it cannot measure."""
    import shiftbench.thermal as thermal

    def refuse(*_: object, **__: object) -> None:
        raise OSError("no sysctl here")

    monkeypatch.setattr(thermal.subprocess, "run", refuse)
    assert thermal.memory_headroom() == 1.0


def test_waiting_for_memory_returns_at_once_when_there_is_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shiftbench.thermal as thermal

    monkeypatch.setattr(thermal, "memory_headroom", lambda: 0.9)
    assert thermal.wait_for_memory(0.25, poll=0.01) == 0.0


def test_waiting_for_memory_gives_up_rather_than_blocking_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shiftbench.thermal as thermal

    monkeypatch.setattr(thermal, "memory_headroom", lambda: 0.01)
    waited = thermal.wait_for_memory(0.25, poll=0.01, patience=0.05)
    assert 0.0 < waited <= 0.06


def test_waiting_for_memory_returns_once_the_machine_frees_some(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shiftbench.thermal as thermal

    readings = iter([0.05, 0.05, 0.6, 0.6])
    monkeypatch.setattr(thermal, "memory_headroom", lambda: next(readings))
    assert 0.0 < thermal.wait_for_memory(0.25, poll=0.01, patience=5.0) < 0.05
