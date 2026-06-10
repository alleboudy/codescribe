"""Token-bucket rate limiter — bursts 20 acquires; assert ≤5/s."""

from __future__ import annotations

import time


def test_token_bucket_caps_burst_to_rate() -> None:
    """20 acquires at 5 rps must take ≥ 3.0 s (i.e. the burst is shaped).

    With capacity = 5 (= rps * 1s window) and refill = 5/s, the bucket
    serves the first 5 instantly, then one token per 200 ms. So 20 acquires
    take ≥ (20 - 5) * 0.2 s = 3.0 s.
    """
    from codescribe_train.rag.sources._ratelimit import TokenBucket

    bucket = TokenBucket(rate_per_second=5.0, capacity=5)
    t0 = time.monotonic()
    for _ in range(20):
        bucket.acquire()
    elapsed = time.monotonic() - t0
    # 20 calls at 5 rps with capacity=5 → ≥ 3.0 s. Allow tiny slack for
    # scheduling jitter on the cold start.
    assert elapsed >= 3.0 - 0.05, f"burst not shaped: {elapsed:.2f}s for 20 acquires"


def test_token_bucket_first_burst_is_instant() -> None:
    """A capacity-sized burst must drain without blocking."""
    from codescribe_train.rag.sources._ratelimit import TokenBucket

    bucket = TokenBucket(rate_per_second=5.0, capacity=5)
    t0 = time.monotonic()
    for _ in range(5):
        bucket.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.1, f"initial burst should be instant, took {elapsed:.3f}s"


def test_token_bucket_acquire_n_tokens() -> None:
    """`acquire(n)` consumes n tokens — burst of 5 then acquire(3) blocks."""
    from codescribe_train.rag.sources._ratelimit import TokenBucket

    bucket = TokenBucket(rate_per_second=10.0, capacity=5)
    # Drain the initial burst.
    for _ in range(5):
        bucket.acquire()
    t0 = time.monotonic()
    bucket.acquire(3)
    elapsed = time.monotonic() - t0
    # At 10 rps we need 3/10 = 0.3 s for 3 tokens.
    assert 0.25 <= elapsed <= 0.5, f"acquire(3) took {elapsed:.3f}s, expected ~0.3s"
