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

    def function(self, name: str, impl=None, *, signature: str = "", doc: str = "", category: str = "",
                 example: str = ""):
        """Register a function. Usable directly or as a decorator."""

        def add(fn):
            self.entries.append(Entry(name, "function", fn, signature or f"{name}(...)", doc, self.name,
                                      category, example))
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


@dataclass
class Registry:
    modules: list[LoadedModule] = field(default_factory=list)

    def namespace(self) -> dict:
        ns: dict = {}
        for mod in self.modules:
            for e in mod.entries:
                ns[e.name] = e.value
        return ns

    def catalog(self) -> dict:
        return {
            "modules": [
                {"name": m.name, "description": m.description, "path": m.path, "error": m.error,
                 "count": len(m.entries)}
                for m in self.modules
            ],
            "entries": [e.describe() for m in self.modules for e in m.entries],
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
        return LoadedModule(name, desc, str(path), api.entries)
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
    return reg
