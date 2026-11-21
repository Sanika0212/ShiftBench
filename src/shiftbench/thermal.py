"""Run inside what the machine doing the scoring can survive.

Written after three restarts on a 16 GB M4 MacBook Air during a sibling project. The panic
log named the cause, and it was not heat: ``userspace watchdog timeout: no successful
checkins from WindowServer in 121 seconds``. The run had exhausted memory, the resulting
swap thrash stalled the display server, and the kernel rebooted the machine. Heat was a
symptom. So memory is bounded first and heat second.

1. ``memory_headroom()`` reports the share of memory free or reclaimable, and
   ``wait_for_memory()`` blocks while the machine is short of it. The scoring loop checks
   between chunks, so another application opening is a pause here rather than a reboot.
2. ``limit_threads()`` caps the thread pools that numerical libraries open by default.
3. ``DutyCycle`` idles between chunks, which is what keeps a fanless machine cool.
4. ``background_qos()`` exists but is off by default: measured on that machine it pinned the
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
MIN_HEADROOM = 0.25


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

    Returns 1.0 where the platform does not report it, so a guard built on this never
    blocks a run on a machine it cannot measure.
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


def wait_for_memory(
    minimum: float = MIN_HEADROOM, poll: float = 15.0, patience: float = 600.0
) -> float:
    """Block while the machine is short of memory; returns the seconds spent waiting.

    After ``patience`` it gives up waiting and returns anyway. A run that waits forever is
    its own kind of failure, and the caller is resumable, so carrying on is safe.
    """
    waited = 0.0
    while memory_headroom() < minimum and waited < patience:
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
