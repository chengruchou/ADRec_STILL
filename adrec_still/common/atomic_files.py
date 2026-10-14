"""Atomic replacement that tolerates short Windows reader sharing windows."""

from __future__ import annotations

import os
import time

REPLACE_RETRY_SECONDS = 2.0


def replace_file(source, destination):
    """Keep the old file intact while a Windows reader briefly denies delete sharing.

    A status reader using Python/PowerShell may hold such a handle. Retry only
    Windows access/sharing errors, for a bounded interval; other errors and a
    persistent lock still propagate. No truncate/delete fallback is permitted.
    The caller owns fsync and retains the sibling temporary on failure.
    """
    deadline = time.monotonic() + REPLACE_RETRY_SECONDS
    delay = 0.005
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            remaining = deadline - time.monotonic()
            if (os.name != "nt" or getattr(error, "winerror", None) not in {5, 32, 33}
                    or remaining <= 0):
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.05)
