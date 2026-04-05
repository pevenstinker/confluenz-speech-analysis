"""
Clock synchronization for confluenz-speech-analysis.

If TIME_SYNC_URL is configured, the offset between the remote server clock and
the local clock is computed once at startup (using an NTP-style midpoint
correction to minimize network latency bias).  All subsequent calls to
get_utc_now() return local time adjusted by that offset.

If TIME_SYNC_URL is blank or the request fails, get_utc_now() returns the
local UTC clock unchanged.
"""
from datetime import datetime, timedelta, timezone

import requests

from config import config

_SERVER_TIME_FMT = "%Y-%m-%d %H:%M:%S.%f"

# None = not yet initialised; timedelta(0) = local clock (no sync / fallback)
_offset: timedelta | None = None


def _init_offset() -> timedelta:
    url = config.time_sync_url.strip()
    if not url:
        return timedelta(0)

    try:
        t0 = datetime.now(timezone.utc)
        response = requests.get(url, timeout=5)
        t1 = datetime.now(timezone.utc)
        response.raise_for_status()

        data = response.json()
        server_dt = datetime.strptime(
            data["serverTime"]["date"], _SERVER_TIME_FMT
        ).replace(tzinfo=timezone.utc)

        # Midpoint correction: assume the response arrived at the midpoint of
        # the round trip, so the true send time is (t0 + t1) / 2.
        midpoint = t0 + (t1 - t0) / 2
        offset = server_dt - midpoint
        print(f"  [clock] Synced with server. Offset: {offset.total_seconds():+.3f}s")
        return offset

    except Exception as exc:
        print(f"  [clock] Warning: time sync failed ({exc}). Using local clock.")
        return timedelta(0)


def get_utc_now() -> datetime:
    """Return the current UTC time, adjusted by the server clock offset if configured."""
    global _offset
    if _offset is None:
        _offset = _init_offset()
    return datetime.now(timezone.utc) + _offset
