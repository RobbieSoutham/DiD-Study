"""Helpers for fetching external datasets with updated Mega TLS handling."""

from __future__ import annotations

import asyncio
from typing import Optional


def _ensure_asyncio_coroutine() -> None:
    """Ensure ``asyncio.coroutine`` exists (removed in Python 3.12).

    The third-party ``mega.py`` client still references ``asyncio.coroutine``.
    We defensively reintroduce it as a no-op decorator so that Mega downloads
    work on modern Python runtimes.
    """

    if not hasattr(asyncio, "coroutine"):
        asyncio.coroutine = lambda func: func  # type: ignore[attr-defined]


def download_from_mega(url: str, dest_path: str) -> Optional[str]:
    """Download a public Mega.nz URL to ``dest_path``.

    Returns the resolved destination path on success, or ``None`` if the Mega
    client is unavailable or raises an exception. Errors are intentionally
    swallowed so callers can fall back to other download strategies or surface
    actionable diagnostics.
    """

    try:
        _ensure_asyncio_coroutine()
        from mega import Mega  # type: ignore

        mega = Mega()
        mega.download_url(url, dest_filename=dest_path)
        return dest_path
    except Exception:
        return None
