"""A derivation: an ordered list of (text, expression) steps ending in a result.

Modules return a Steps object; the evaluator renders each step as text plus LaTeX
and the final result as the cell's value.
"""
from __future__ import annotations

import sympy as sp


class Steps:
    def __init__(self, result, steps: list[tuple[str, object]], title: str = ""):
        self.result = result
        self.steps = list(steps)
        self.title = title

    def add(self, text: str, expr=None):
        self.steps.append((text, expr))
        return self

    def __repr__(self):
        return f"Steps({len(self.steps)} steps -> {self.result})"

    @property
    def free_symbols(self):
        return getattr(self.result, "free_symbols", set())

    def describe(self) -> list[dict]:
        out = []
        for text, expr in self.steps:
            item = {"text": text}
            if expr is not None:
                try:
                    item["latex"] = sp.latex(expr) if isinstance(expr, sp.Basic) else str(expr)
                    item["plain"] = str(expr)
                except Exception:  # noqa: BLE001
                    item["latex"] = r"\text{" + str(expr) + "}"
                    item["plain"] = str(expr)
            out.append(item)
        return out
