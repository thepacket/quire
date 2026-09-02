"""The built-in module: units, constants and the standard math functions, split by domain."""
from __future__ import annotations

from .builtin import algebra, basics, calculus, linalg, numbers, recognize, special, transforms

NAME = "core"
DESCRIPTION = "Units, constants, algebra, calculus, equations, matrices, transforms."


def register(api):
    for part in (basics, algebra, calculus, linalg, numbers, special, transforms, recognize):
        part.register(api)
