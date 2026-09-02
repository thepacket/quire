"""Backend fallbacks registered by modules, consulted by core operations when SymPy gives up."""
from __future__ import annotations

_FALLBACKS: dict[str, list] = {}
context: dict = {}  # per-evaluation information for backends (e.g. "bounds": {name: [(op, value)]})


def install(table: dict[str, list]) -> None:
    _FALLBACKS.clear()
    _FALLBACKS.update(table)


def run(operation: str, *args, accept=None, **kwargs):
    """Return the first backend answer that is not None (and passes ``accept`` if given). Errors are swallowed."""
    for fn in _FALLBACKS.get(operation, []):
        try:
            res = fn(*args, **kwargs)
        except Exception:  # noqa: BLE001 - a broken backend must not break the worksheet
            continue
        if res is not None and (accept is None or accept(res)):
            return res
    return None


def available(operation: str) -> bool:
    return bool(_FALLBACKS.get(operation))


def run_all(operation: str, *args, accept=None, **kwargs) -> list:
    """Every backend answer that is not None (and passes ``accept``), in priority order."""
    out = []
    for fn in _FALLBACKS.get(operation, []):
        try:
            res = fn(*args, **kwargs)
        except Exception:  # noqa: BLE001
            continue
        if res is not None and (accept is None or accept(res)):
            out.append(res)
    return out
