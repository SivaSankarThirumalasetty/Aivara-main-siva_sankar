"""
insight/cache.py — facts-hash -> LLM response cache.

Keys the LLM response by a hash of the facts packet JSON (+ model name +
prompt version), so re-uploading the same data or re-viewing a tab never
triggers a duplicate call. This is also the module export/pptx_builder.py
reads from directly — it must never import insight/client.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading

from config import INSIGHT_CACHE_PATH

_lock = threading.Lock()


def _hash_key(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _read_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_cache(path: str, cache: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp_path, path)


def get(cache_key_material: str, path: str = INSIGHT_CACHE_PATH) -> dict | None:
    """Returns the cached insight dict (headline/driver/action/raw_text) or
    None if not cached."""
    key = _hash_key(cache_key_material)
    with _lock:
        cache = _read_cache(path)
        return cache.get(key)


def set(cache_key_material: str, value: dict, path: str = INSIGHT_CACHE_PATH) -> None:
    key = _hash_key(cache_key_material)
    with _lock:
        cache = _read_cache(path)
        cache[key] = value
        _write_cache(path, cache)


_view_cache: dict[str, dict] = {}


def get_for_view(view_name: str, session_id: str | None = None) -> dict | None:
    """Convenience lookup used by export/pptx_builder.py. Scoped to session state
    or in-memory cache to prevent cross-user data leakage across sessions.
    Never persisted to the shared disk cache."""
    if session_id:
        with _lock:
            return _view_cache.get(f"{session_id}::{view_name}")

    try:
        import streamlit as st

        if hasattr(st, "session_state"):
            if "_view_insights" in st.session_state:
                return st.session_state["_view_insights"].get(view_name)
            # In Streamlit runtime, do NOT fall back to global un-scoped _view_cache
            return None
    except Exception:
        pass

    with _lock:
        return _view_cache.get(view_name)


def set_for_view(view_name: str, value: dict, session_id: str | None = None) -> None:
    if session_id:
        with _lock:
            _view_cache[f"{session_id}::{view_name}"] = value
        return

    try:
        import streamlit as st

        if hasattr(st, "session_state"):
            st.session_state.setdefault("_view_insights", {})[view_name] = value
            return
    except Exception:
        pass

    with _lock:
        _view_cache[view_name] = value


def clear_view_cache(session_id: str | None = None) -> None:
    """Clear in-memory view cache."""
    with _lock:
        if session_id:
            keys_to_del = [k for k in _view_cache if k.startswith(f"{session_id}::")]
            for k in keys_to_del:
                del _view_cache[k]
        else:
            _view_cache.clear()

