from __future__ import annotations

from time import time, sleep
from typing import Any, Dict, Optional, Tuple
import threading


class TTLCache:
    """Simple in-memory TTL cache with optional background auto-purge.

    - Capacity-bounded; evicts entries closest to expiry first when over capacity.
    - Thread-safe using a simple lock.
    - Optional background thread purges expired entries every N seconds so
    memory is reclaimed even when there are no reads.
    """

    def __init__(
        self,
        max_items: int = 256,
        auto_purge_interval_seconds: Optional[float] = 15.0,
    ) -> None:
        self._data: Dict[Any, Tuple[float, Any]] = {}
        self._max = max_items
        self._lock = threading.Lock()
        self._stop_flag = False
        self._auto_interval = auto_purge_interval_seconds
        self._thread: Optional[threading.Thread] = None
        if self._auto_interval and self._auto_interval > 0:
            self._thread = threading.Thread(target=self._auto_purge_loop, daemon=True)
            self._thread.start()

    def _auto_purge_loop(self) -> None:
        try:
            while not self._stop_flag:
                sleep(self._auto_interval or 15.0)
                self._purge()
        except Exception:
            # best-effort; never raise from daemon thread
            pass

    def stop(self) -> None:
        self._stop_flag = True

    def _purge(self) -> None:
        now = time()
        with self._lock:
            # Remove expired
            expired = [k for k, (exp, _) in self._data.items() if exp < now]
            for k in expired:
                self._data.pop(k, None)
            # Enforce capacity
            if len(self._data) > self._max:
                over = len(self._data) - self._max
                for k, _ in sorted(self._data.items(), key=lambda kv: kv[1][0])[:over]:
                    self._data.pop(k, None)

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            exp, val = item
            if exp < time():
                # expired
                self._data.pop(key, None)
                return None
            return val

    def set(self, key: Any, value: Any, ttl_seconds: float) -> None:
        # Store even None values for negative-caching
        with self._lock:
            self._data[key] = (time() + float(ttl_seconds), value)
        # opportunistic purge
        self._purge()

    def clear(self, key: Any) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear_all(self) -> None:
        with self._lock:
            self._data.clear()


# Shared caches and TTLs used by services
CACHE_TTL_OK = 30.0             # 30s for successful Jira fetch (demo-friendly)
CACHE_TTL_ERROR = 30.0          # 30s negative cache on Jira errors/misses
CACHE_TTL_SIMILAR_OK = 180.0    # 3 minutes for similar results
CACHE_TTL_SIMILAR_ERROR = 45.0  # short on empty/error similar results

JIRA_TICKET_CACHE = TTLCache(max_items=512, auto_purge_interval_seconds=15.0)
JIRA_SIMILAR_CACHE = TTLCache(max_items=1024, auto_purge_interval_seconds=15.0)
