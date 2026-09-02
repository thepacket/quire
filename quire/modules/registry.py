"""Module system.

A module is a directory containing ``module.py`` with a ``register(api)``
function. Through ``api`` it adds functions, constants and units that become
available in every worksheet, and it describes them so the reference panel can
show them. Modules never see the UI; they only extend the math namespace.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Entry:
    name: str
    kind: str  # function | constant | unit
    value: object
    signature: str = ""
    doc: str = ""
    module: str = ""
    category: str = ""
    example: str = ""
    hidden: bool = False

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "signature": self.signature,
            "doc": self.doc,
            "module": self.module,
            "category": self.category,
            "example": self.example,
        }


class ModuleAPI:
    """The object handed to a module's ``register`` function."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.entries: list[Entry] = []
        self.fallbacks: dict[str, list] = {}

    def fallback(self, operation: str, fn, priority: int = 50):
        """Register a backend for an operation ("integrate", "simplify", "limit", "sum").

        Core functions call the backends in priority order (lower first) whenever
        SymPy gives up (an unevaluated Integral, an unchanged expression). ``fn``
        receives the same arguments as the core operation and returns a sympy
        object or None to pass.
        """
        self.fallbacks.setdefault(operation, []).append((priority, fn))

    def function(self, name: str, impl=None, *, signature: str = "", doc: str = "", category: str = "",
                 example: str = "", hidden: bool = False):
        """Register a function. Usable directly or as a decorator. Hidden entries stay out of the catalog."""

        def add(fn):
            self.entries.append(Entry(name, "function", fn, signature or f"{name}(...)", doc, self.name,
                                      category, example, hidden))
            return fn

        return add(impl) if impl is not None else add

    def constant(self, name: str, value, *, doc: str = "", category: str = "", example: str = ""):
        self.entries.append(Entry(name, "constant", value, name, doc, self.name, category, example))

    def unit(self, name: str, quantity, *, doc: str = "", category: str = "", example: str = ""):
        self.entries.append(Entry(name, "unit", quantity, name, doc, self.name, category, example))


@dataclass
class LoadedModule:
    name: str
    description: str
    path: str
    entries: list[Entry]
    error: str | None = None
    fallbacks: dict = field(default_factory=dict)


@dataclass
class Registry:
    modules: list[LoadedModule] = field(default_factory=list)

    def namespace(self) -> dict:
        ns: dict = {}
        for mod in self.modules:
            for e in mod.entries:
                ns[e.name] = e.value
        return ns

    def conflicts(self) -> list[str]:
        """Names defined by more than one module (the later module wins)."""
        seen: dict[str, str] = {}
        out = []
        for mod in self.modules:
            for e in mod.entries:
                if e.kind == "unit" and len(e.name) == 1:
                    continue  # single-letter units only count directly after a number; no clash with names
                if e.name in seen and seen[e.name] != mod.name:
                    out.append(f"'{e.name}' is defined by both {seen[e.name]} and {mod.name}; {mod.name} wins")
                seen[e.name] = mod.name
        return out

    def catalog(self) -> dict:
        return {
            "modules": [
                {"name": m.name, "description": m.description, "path": m.path, "error": m.error,
                 "count": len(m.entries)}
                for m in self.modules
            ],
            "entries": [e.describe() for m in self.modules for e in m.entries if not e.hidden],
            "conflicts": self.conflicts(),
        }


def _load_from_file(path: Path, fallback_name: str) -> LoadedModule:
    spec = importlib.util.spec_from_file_location(f"quire_module_{fallback_name}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        name = getattr(module, "NAME", fallback_name)
        desc = getattr(module, "DESCRIPTION", "")
        api = ModuleAPI(name, desc)
        module.register(api)
        return LoadedModule(name, desc, str(path), api.entries, fallbacks=api.fallbacks)
    except Exception:
        err = traceback.format_exc(limit=3)
        return LoadedModule(fallback_name, "", str(path), [], error=err)


def load_registry(module_dirs: list[Path]) -> Registry:
    """Load the built-in core module, then every ``*/module.py`` in module_dirs."""
    from . import core

    api = ModuleAPI(core.NAME, core.DESCRIPTION)
    core.register(api)
    reg = Registry([LoadedModule(core.NAME, core.DESCRIPTION, core.__file__, api.entries)])
    for d in module_dirs:
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            mp = sub / "module.py"
            if sub.is_dir() and mp.is_file():
                reg.modules.append(_load_from_file(mp, sub.name))
    from . import hooks

    ops = {op for m in reg.modules for op in m.fallbacks}
    hooks.install({op: [fn for _, fn in sorted((pf for m in reg.modules for pf in m.fallbacks.get(op, [])),
                                                key=lambda pf: pf[0])] for op in ops})
    return reg
