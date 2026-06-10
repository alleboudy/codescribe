"""Shared token-bucket rate limiter for sources.

GitHub's REST authenticated quota is 5000/h; the AGENTS.md ruleset caps the
sources package at 5 rps (3000/h) for headroom. This module provides the
single ``TokenBucket`` shared across all ``GitHubSource`` methods.
"""

from __future__ import annotations

import threading
import time


class TokenBucket:
    """Thread-safe blocking token bucket.

    Tokens refill at ``rate_per_second`` up to ``capacity``. ``acquire(n)``
    blocks the calling thread until ``n`` tokens are available, then consumes
    them. Time is measured with ``time.monotonic`` so wall-clock skew does
    not corrupt the bucket.
    """

    def __init__(self, rate_per_second: float, capacity: int | None = None) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        self._rate = float(rate_per_second)
        # Default capacity = one second of refill so a fresh bucket can absorb
        # a 1-second burst without shaping.
        self._capacity = float(capacity if capacity is not None else max(1, int(rate_per_second)))
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        delta = now - self._last_refill
        if delta > 0:
            self._tokens = min(self._capacity, self._tokens + delta * self._rate)
            self._last_refill = now

    def acquire(self, n: int = 1) -> None:
        """Block until ``n`` tokens can be consumed, then consume them."""
        if n <= 0:
            raise ValueError("n must be > 0")
        if n > self._capacity:
            raise ValueError(
                f"requested {n} tokens but bucket capacity is {self._capacity}"
            )
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= n:
                    self._tokens -= n
                    return
                missing = n - self._tokens
                wait = missing / self._rate
            time.sleep(wait)
