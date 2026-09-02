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
        self.plot_kinds: dict[str, dict] = {}

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

    def plot_kind(self, name: str, fn, *, label: str, f1: str, f2: str | None = None, f3: str | None = None,
                  var: str | None = None, range: str | None = None, yrange: bool = False, ph1: str = "",
                  ph2: str = "", ph3: str = "", renderer: str = "svg", annot: bool = False, doc: str = "",
                  samples: str = "400"):
        """Add a plot kind to the plot cell's menu.

        ``fn(cell, env, ev)`` receives the cell fields (exprs, expr2, expr3, var, xmin, xmax, ymin, ymax,
        samples), the worksheet environment and the evaluator, and returns {"series": [...], ...} in the
        same shape as the core kinds (see quire.engine.plotting, whose helpers modules may use). The other
        arguments label the form fields, as in quire.engine.plotting.register_kind.
        """
        self.plot_kinds[name] = dict(fn=fn, name=name, label=label, f1=f1, f2=f2, f3=f3, var=var, range=range,
                                     yrange=yrange, ph1=ph1, ph2=ph2, ph3=ph3, renderer=renderer, annot=annot,
                                     module=self.name, doc=doc, samples=samples)

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
    plot_kinds: dict = field(default_factory=dict)


# Language constructs, shown first in the reference panel. (name, signature, doc, example)
SYNTAX = [
    ("definition", "x = 5", "define a value for every cell below", "x = 5"),
    ("function", "f(x) = x^2", "define a function", "f(x) = x^2 + 1"),
    ("units", "3 m/s^2", "a number followed by units; 2 kg m/s^2, 4.7 kohm", "3 m/s^2"),
    ("conversion", "expr -> km/hr", "convert a result to other units (-> deg for angles)", "-> km/hr"),
    ("equation", "a == b", "an equation, for solve, nsolve, dsolve", "solve(x^2 == 4, x)"),
    ("assume", "assume x > 0", "assumptions: > 0, >= 0, != 0, real, integer, positive integer, s > 1", "assume x > 0"),
    ("digits", "digits 10", "significant digits shown in the cells below (default 6)", "digits 10"),
    ("prime", "f'(x)", "derivative of a defined function; f''(x) for the second", "f'(2)"),
    ("sequence", "a[n]", "a term of a sequence, for rsolve", "rsolve(a[n+1] == 2 a[n], a, n, 1)"),
    ("power", "x^2", "^ is the power; 2 pi, sin x, n! also work", "x^2"),
    ("list, matrix", "[1, 2, 3]", "lists in brackets; matrix([[1, 2], [3, 4]]) for matrices", "[1, 2, 3]"),
    ("several lines", "a = 1\nb = a + 1", "lines in one cell are evaluated in order", ""),
    ("integral sign", "∫_0^1 x^2 dx", "∫ with optional _lower^upper bounds; ∫∫ for double integrals", "∫_0^1 x^2 dx"),
    ("sum sign", "Σ_{k=1}^{n} k^2", "Σ or Π with _{k=a}^{b} bounds", "Σ_{k=1}^{n} k^2"),
    ("d/dx", "d/dx sin(x)", "derivative in paper notation; d²/dx² for the second", "d/dx sin(x)"),
    ("symbols", "θ, x², √2, ≤, ×", "Greek letters, superscripts and subscripts, √, ≤ ≥ ≠ × · are read as written", "√2 × θ²"),
    ("LaTeX", "\\frac{1}{2} \\sqrt{x}", "\\frac, \\sqrt, \\int, \\sum, \\alpha … are accepted", "\\int_0^1 \\frac{1}{1+x^2} dx"),
]


@dataclass
class Registry:
    modules: list[LoadedModule] = field(default_factory=list)

    def namespace(self) -> dict:
        ns: dict = {}
        for mod in self.modules:
            for e in mod.entries:
                ns[e.name] = e.value
        return ns

    def plot_kinds(self) -> dict:
        """Plot kinds added by modules, name -> descriptor (with "fn")."""
        out: dict = {}
        for mod in self.modules:
            out.update(mod.plot_kinds)
        return out

    def conflicts(self) -> list[str]:
        """Names defined by more than one module (the later module wins)."""
        from ..engine.plotting import KINDS

        seen: dict[str, str] = {}
        out = []
        kinds: dict[str, str] = {k: "core" for k in KINDS}
        for mod in self.modules:
            for k in mod.plot_kinds:
                if k in kinds:
                    out.append(f"plot kind '{k}' is defined by both {kinds[k]} and {mod.name}; {mod.name} wins")
                kinds[k] = mod.name
        for mod in self.modules:
            for e in mod.entries:
                if e.kind == "unit" and len(e.name) == 1:
                    continue  # single-letter units only count directly after a number; no clash with names
                if e.name in seen and seen[e.name] != mod.name:
                    out.append(f"'{e.name}' is defined by both {seen[e.name]} and {mod.name}; {mod.name} wins")
                seen[e.name] = mod.name
        return out

    def catalog(self) -> dict:
        from ..engine.plotting import describe_kinds

        kinds = describe_kinds(self.plot_kinds())
        return {
            "plot_kinds": kinds,
            "modules": [
                {"name": m.name, "description": m.description, "path": m.path, "error": m.error,
                 "count": len(m.entries)}
                for m in self.modules
            ],
            "entries": [{"name": n, "kind": "syntax", "signature": sig, "doc": doc, "module": "core",
                         "category": "Syntax", "example": ex} for n, sig, doc, ex in SYNTAX]
                       + [e.describe() for m in self.modules for e in m.entries if not e.hidden]
                       + [{"name": k["label"], "kind": "plot", "signature": k["label"], "doc": k["doc"],
                           "module": k["module"], "category": "Plots", "example": k["name"]} for k in kinds],
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
        return LoadedModule(name, desc, str(path), api.entries, fallbacks=api.fallbacks, plot_kinds=api.plot_kinds)
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
