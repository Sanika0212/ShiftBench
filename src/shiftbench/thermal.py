"""Run inside what the machine doing the scoring can survive.

Written after three restarts on a 16 GB M4 MacBook Air during a sibling project. The panic
log named the cause, and it was not heat: ``userspace watchdog timeout: no successful
checkins from WindowServer in 121 seconds``. The run had exhausted memory, the resulting
swap thrash stalled the display server, and the kernel rebooted the machine. Heat was a
symptom. So memory is bounded first and heat second.

The first version of this gate counted free pages and waited below 25% of them. That gate
cannot open on this machine: free plus inactive plus speculative pages sit near 24% whatever
is happening, so the condition held with memory to spare and parked a sibling run for 43
minutes with 5.5 seconds of CPU to show for it. Free pages are not a danger signal on a
system that keeps its memory full on purpose. The kernel publishes two numbers that are:

1. ``kern.memorystatus_level`` is the share Jetsam considers available, and Jetsam is what
   was killing processes on the reboot night, so this is the number that matters.
2. ``kern.memorystatus_vm_pressure_level`` is 1 normal, 2 warning, 4 critical. Everyday use
   of this laptop sits at 2, so only 4 is worth stopping for.

So ``wait_for_memory()`` waits when availability is genuinely low or pressure is critical,
and its patience is a minute rather than ten, because a gate that parks a run is as bad as
no gate at all. ``memory_headroom()`` is kept, for reporting only. Around that:

3. ``limit_threads()`` caps the thread pools that numerical libraries open by default.
4. ``DutyCycle`` idles between chunks, which is what keeps a fanless machine cool.
5. ``background_qos()`` exists but is off by default: measured on that machine it pinned the
   work to the efficiency cores and cost about ten times the run time, which is more total
   heat for the same work.

The cost of all this is wall clock, and a run that can be stopped and resumed (see the
per-claim cache in ``runner``) makes that cost affordable: stop it whenever, start it again
later, nothing is lost.
"""

from __future__ import annotations

import os
import subprocess
import time

DEFAULT_THREADS = 2
# Jetsam's own availability figure, below which a run should wait. This laptop idles around
# 40%, and the night it rebooted Jetsam was killing applications outright.
MIN_AVAILABLE = 0.18
# 1 normal, 2 warning, 4 critical. Ordinary use of this machine sits at 2.
CRITICAL_PRESSURE = 4


def limit_threads(threads: int = DEFAULT_THREADS) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, str(threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def memory_headroom() -> float:
    """Share of physical memory that is free or reclaimable, from 0 to 1.

    For reporting, not for gating: see the note at the top of this module. Returns 1.0
    where the platform does not report it.
    """
    try:
        total = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return 1.0
    page = 16384
    counts: dict[str, int] = {}
    for line in out.splitlines():
        if line.startswith("Mach Virtual Memory Statistics"):
            if "page size of" in line:
                page = int(line.split("page size of")[1].split("bytes")[0].strip())
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            digits = value.strip().rstrip(".")
            if digits.isdigit():
                counts[key.strip()] = int(digits)
    free = sum(counts.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative"))
    return min(1.0, free * page / total) if total else 1.0


def _sysctl_int(name: str) -> int | None:
    """One integer sysctl, or None where the platform does not have it."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out) if out.lstrip("-").isdigit() else None


def available_memory() -> float:
    """The share of memory Jetsam considers available, from 0 to 1.

    Returns 1.0 where it cannot be read, so a guard built on it never blocks a run on a
    machine it cannot measure.
    """
    level = _sysctl_int("kern.memorystatus_level")
    return 1.0 if level is None else max(0.0, min(1.0, level / 100))


def pressure_level() -> int:
    """1 normal, 2 warning, 4 critical, from ``kern.memorystatus_vm_pressure_level``."""
    level = _sysctl_int("kern.memorystatus_vm_pressure_level")
    return 1 if level is None else level


def memory_is_tight(minimum: float = MIN_AVAILABLE, critical: int = CRITICAL_PRESSURE) -> bool:
    """Whether the machine is in real trouble, as opposed to merely busy."""
    return available_memory() < minimum or pressure_level() >= critical


def wait_for_memory(
    minimum: float = MIN_AVAILABLE,
    poll: float = 5.0,
    patience: float = 60.0,
    critical: int = CRITICAL_PRESSURE,
) -> float:
    """Block while the machine is genuinely short of memory; returns the seconds waited.

    Patience is deliberately short. The point is to pace a run around a burst of pressure,
    not to park it. After ``patience`` it returns and lets the run continue; the caller is
    resumable, so a run that carries on and is stopped later loses nothing.
    """
    waited = 0.0
    while memory_is_tight(minimum, critical) and waited < patience:
        time.sleep(poll)
        waited += poll
    return waited


def background_qos() -> bool:
    """Ask macOS for background scheduling. True when it was accepted.

    Off by default in the scoring loop: see the note at the top of this module.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.dylib", use_errno=True)
        return bool(libc.setpriority(4, 0, 0x1000) == 0)  # PRIO_DARWIN_PROCESS, PRIO_DARWIN_BG
    except (OSError, AttributeError):
        return False


class DutyCycle:
    """Idle between chunks of work so the machine stays cool enough to finish the run."""

    def __init__(self, duty: float = 0.4, max_pause: float = 20.0) -> None:
        if not 0 < duty <= 1:
            raise ValueError("duty must be in (0, 1]")
        self.duty, self.max_pause = duty, max_pause
        self.started = time.monotonic()
        self.worked = self.paused = 0.0

    def __enter__(self) -> DutyCycle:
        self.started = time.monotonic()
        return self

    def __exit__(self, *_: object) -> None:
        busy = time.monotonic() - self.started
        self.worked += busy
        if self.duty >= 1:
            return
        pause = min(busy * (1 - self.duty) / self.duty, self.max_pause)
        if pause > 0:
            time.sleep(pause)
            self.paused += pause

    @property
    def share(self) -> float:
        total = self.worked + self.paused
        return self.worked / total if total else 1.0
