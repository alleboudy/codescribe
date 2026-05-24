from __future__ import annotations

import threading
import time

from codescribe_rag.rag.sources._ratelimit import TokenBucket


def test_initial_burst_immediate():
    tb = TokenBucket(rate_per_second=5.0, burst=5)
    start = time.monotonic()
    for _ in range(5):
        tb.acquire()
    assert time.monotonic() - start < 0.1


def test_blocks_until_refilled():
    tb = TokenBucket(rate_per_second=10.0, burst=1)
    tb.acquire()
    start = time.monotonic()
    tb.acquire()
    assert time.monotonic() - start >= 0.08  # ~1/10s, allow scheduler slack


def test_thread_safe_no_overdraw():
    tb = TokenBucket(rate_per_second=50.0, burst=1)
    start = time.monotonic()

    def worker():
        for _ in range(10):
            tb.acquire()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 40 acquires at 50/s with burst 1 ~= >= 39/50 s
    assert time.monotonic() - start >= 0.6
