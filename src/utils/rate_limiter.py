"""Adaptive rate limiter for Gemini API with multi-key rotation.

Features:
- Proactive pacing: sleeps BEFORE sending requests to stay under RPM limit
- Multi-key rotation: cycles through multiple API keys to multiply throughput
- Sliding window tracking: monitors actual usage per key
- Thread-safe: works with both sync and async callers
"""

import asyncio
import logging
import time
import threading
from collections import deque
from typing import List, Optional

logger = logging.getLogger(__name__)


class GeminiRateLimiter:
    """Rate limiter with proactive pacing and multi-key rotation.

    Args:
        api_keys: List of Gemini API keys to rotate through.
        rpm_limit: Requests per minute per key (default: 15 for free tier).
        cooldown_seconds: Extra cooldown added when a 429 is reported.
    """

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        rpm_limit: int = 15,
        cooldown_seconds: float = 60.0,
    ):
        if not api_keys:
            raise ValueError("At least one API key is required")

        self._keys = [k.strip() for k in api_keys if k.strip()]
        if not self._keys:
            raise ValueError("No valid API keys provided")

        self._rpm_limit = rpm_limit
        self._cooldown_seconds = cooldown_seconds

        # Per-key sliding window of request timestamps
        self._windows: dict[str, deque] = {k: deque() for k in self._keys}
        # Per-key cooldown until timestamp (set when a 429 is hit)
        self._cooldowns: dict[str, float] = {k: 0.0 for k in self._keys}
        # Round-robin index
        self._next_index = 0

        self._lock = threading.Lock()

        # Minimum interval between requests per key
        self._min_interval = 60.0 / max(rpm_limit, 1)

        logger.info(
            f"RateLimiter initialized: {len(self._keys)} key(s), "
            f"{rpm_limit} RPM/key, "
            f"effective throughput: ~{rpm_limit * len(self._keys)} RPM"
        )

    @property
    def total_rpm(self) -> int:
        """Total effective RPM across all keys."""
        return self._rpm_limit * len(self._keys)

    def _purge_old(self, key: str) -> None:
        """Remove timestamps older than 60 seconds from the sliding window."""
        cutoff = time.monotonic() - 60.0
        window = self._windows[key]
        while window and window[0] < cutoff:
            window.popleft()

    def _pick_key(self) -> tuple[str, float]:
        """Pick the best available key and return (key, wait_seconds).

        Strategy: round-robin, but skip keys in cooldown. If all keys are
        in cooldown, return the one with the shortest remaining cooldown.
        """
        now = time.monotonic()
        best_key = None
        best_wait = float("inf")

        for _ in range(len(self._keys)):
            idx = self._next_index % len(self._keys)
            self._next_index += 1
            key = self._keys[idx]

            # Check cooldown
            cooldown_remaining = max(0.0, self._cooldowns[key] - now)
            if cooldown_remaining > 0:
                if cooldown_remaining < best_wait:
                    best_wait = cooldown_remaining
                    best_key = key
                continue

            # Check RPM window
            self._purge_old(key)
            window = self._windows[key]

            if len(window) < self._rpm_limit:
                # Key has capacity — check min interval
                if window:
                    elapsed = now - window[-1]
                    if elapsed < self._min_interval:
                        wait = self._min_interval - elapsed
                    else:
                        wait = 0.0
                else:
                    wait = 0.0

                if wait < best_wait:
                    best_wait = wait
                    best_key = key

                if wait == 0.0:
                    break  # Found an immediately available key
            else:
                # Key is at capacity — wait until oldest request expires
                wait = 60.0 - (now - window[0]) + 0.1
                if wait < best_wait:
                    best_wait = wait
                    best_key = key

        if best_key is None:
            best_key = self._keys[0]
            best_wait = self._min_interval

        return best_key, max(0.0, best_wait)

    def acquire_sync(self) -> str:
        """Block until a key is available, return the API key to use.

        Thread-safe. Call this before each Gemini API request.
        """
        while True:
            with self._lock:
                key, wait = self._pick_key()
                if wait <= 0:
                    self._windows[key].append(time.monotonic())
                    return key

            if wait > 0:
                logger.debug(f"Rate limiter: waiting {wait:.1f}s before next request")
                time.sleep(wait)

    async def acquire(self) -> str:
        """Async version of acquire_sync.

        Await this before each Gemini API request.
        """
        while True:
            with self._lock:
                key, wait = self._pick_key()
                if wait <= 0:
                    self._windows[key].append(time.monotonic())
                    return key

            if wait > 0:
                logger.debug(f"Rate limiter: waiting {wait:.1f}s before next request")
                await asyncio.sleep(wait)

    def report_rate_limit(self, key: str) -> None:
        """Report that a 429 was received for this key.

        Puts the key in cooldown so other keys are preferred.
        """
        with self._lock:
            self._cooldowns[key] = time.monotonic() + self._cooldown_seconds
            logger.warning(
                f"Rate limit hit for key ...{key[-6:]}, "
                f"cooldown {self._cooldown_seconds}s"
            )

    def report_success(self, key: str) -> None:
        """Report a successful request (clears any residual cooldown)."""
        with self._lock:
            # Don't clear cooldown — let it expire naturally
            pass

    def stats(self) -> dict:
        """Return current rate limiter statistics."""
        now = time.monotonic()
        with self._lock:
            stats = {}
            for key in self._keys:
                self._purge_old(key)
                stats[f"...{key[-6:]}"] = {
                    "requests_last_60s": len(self._windows[key]),
                    "in_cooldown": self._cooldowns[key] > now,
                    "cooldown_remaining": max(0.0, self._cooldowns[key] - now),
                }
            return stats


# Module-level singleton — initialized lazily
_global_limiter: Optional[GeminiRateLimiter] = None
_global_lock = threading.Lock()


def get_rate_limiter(
    api_keys: Optional[List[str]] = None,
    rpm_limit: Optional[int] = None,
) -> GeminiRateLimiter:
    """Get or create the global rate limiter singleton.

    On first call, creates the limiter with the provided keys/limits.
    Subsequent calls return the same instance.
    """
    global _global_limiter
    with _global_lock:
        if _global_limiter is None:
            from ..config import GEMINI_API_KEYS, GEMINI_RPM_LIMIT

            keys = api_keys or GEMINI_API_KEYS
            limit = rpm_limit or GEMINI_RPM_LIMIT

            _global_limiter = GeminiRateLimiter(
                api_keys=keys,
                rpm_limit=limit,
            )
        return _global_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter (useful for testing)."""
    global _global_limiter
    with _global_lock:
        _global_limiter = None
