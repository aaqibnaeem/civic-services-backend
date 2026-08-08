"""A minimal circuit breaker for the DeepSeek tier.

The problem it solves
---------------------
The API runs on a Render free instance: 0.1 CPU, 512 MB, a handful of workers.
The LLM retry policy has a total budget of ~45 s. If DeepSeek goes down (its
published failure mode is chunky — 9 incidents in 90 days, median ~1h05m), then
*every* analysis request spends 45 s failing before falling back. A dozen
concurrent submissions is enough to occupy every worker and take the whole app
down, including the pages that have nothing to do with AI.

So after ``failure_threshold`` consecutive failures the breaker opens and the LLM
tier is skipped entirely for ``reset_seconds``. Analysis still succeeds — it just
comes from the ML tier instead, which costs 0.1 ms. One probe request is allowed
through when the timer expires (half-open); if it succeeds the breaker closes.

    CLOSED  --failures >= threshold-->  OPEN
    OPEN    --reset_seconds elapsed-->  HALF_OPEN
    HALF_OPEN --probe ok-->  CLOSED
    HALF_OPEN --probe fails-->  OPEN (timer restarts)

Deliberately in-process and not distributed. With one or two Render workers a
shared Redis breaker would add a network dependency to the component whose entire
job is surviving network failure.
"""

from __future__ import annotations

import threading
import time
from typing import Literal

State = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    """Thread-safe consecutive-failure breaker.

    Args:
        name: identifier used in health output and logs.
        failure_threshold: consecutive failures that trip the breaker.
        reset_seconds: how long to stay open before allowing a probe.
    """

    def __init__(self, name: str = "deepseek", failure_threshold: int = 3,
                 reset_seconds: float = 60.0) -> None:
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.reset_seconds = max(1.0, reset_seconds)

        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_in_flight = False

        # cumulative counters, for /ai/health
        self._total_success = 0
        self._total_failure = 0
        self._total_short_circuited = 0
        self._last_error: str | None = None
        self._last_error_at: float | None = None
        self._last_success_at: float | None = None

    # -- state ---------------------------------------------------------------

    @property
    def state(self) -> State:
        with self._lock:
            return self._state_locked()

    def _state_locked(self) -> State:
        if self._opened_at is None:
            return "closed"
        if (time.monotonic() - self._opened_at) >= self.reset_seconds:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        """True if a call may proceed. Reserves the single half-open probe slot."""
        with self._lock:
            state = self._state_locked()
            if state == "closed":
                return True
            if state == "open":
                self._total_short_circuited += 1
                return False
            # half_open: let exactly one probe through at a time
            if self._half_open_in_flight:
                self._total_short_circuited += 1
                return False
            self._half_open_in_flight = True
            return True

    def retry_after_seconds(self) -> float:
        """Seconds until the breaker will next allow a probe (0 if it already does)."""
        with self._lock:
            if self._opened_at is None:
                return 0.0
            remaining = self.reset_seconds - (time.monotonic() - self._opened_at)
            return max(0.0, round(remaining, 1))

    # -- outcomes ------------------------------------------------------------

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None
            self._half_open_in_flight = False
            self._total_success += 1
            self._last_success_at = time.time()

    def record_failure(self, error: str | None = None) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._total_failure += 1
            if error:
                self._last_error = error[:500]
                self._last_error_at = time.time()
            was_probe = self._half_open_in_flight
            self._half_open_in_flight = False
            if was_probe or self._consecutive_failures >= self.failure_threshold:
                # a failed probe re-opens immediately and restarts the timer
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        """Force closed. Used by tests and by an operator 'try again now' action."""
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None
            self._half_open_in_flight = False

    # -- introspection -------------------------------------------------------

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def snapshot(self) -> dict:
        """Diagnostics for ``GET /ai/health``."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state_locked(),
                "consecutive_failures": self._consecutive_failures,
                "failure_threshold": self.failure_threshold,
                "reset_seconds": self.reset_seconds,
                "total_success": self._total_success,
                "total_failure": self._total_failure,
                "total_short_circuited": self._total_short_circuited,
                "last_error": self._last_error,
                "last_error_at": self._last_error_at,
                "last_success_at": self._last_success_at,
            }


#: Process-wide breaker guarding the DeepSeek tier.
llm_breaker = CircuitBreaker(name="deepseek", failure_threshold=3, reset_seconds=60.0)
