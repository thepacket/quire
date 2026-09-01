# Quire

A reactive, unit-aware math worksheet. You write math, not code:

```
m = 2 kg
a = 3 m/s^2
F = m a -> N            →  F = 6 N
f(x) = x^3 - 2 x + 1
solve(f(x) == 0, x)
integrate(f(x), x, 0, 2)
```

Definitions flow top to bottom. Change a number and every cell below it updates.
Units travel with the math: `y(t) = v_0 t - 1/2 g t^2` evaluated at `y(1 s)` comes
out in meters, and `2 km + 3 s` is an error that says why. Anything you leave undefined
stays symbolic, so `diff(1/2 M v^2, v)` gives `M v`.

## Run it

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m quire
```

That starts a local server on http://127.0.0.1:8765 and opens it in your browser.
Worksheets are saved as JSON in `./worksheets` (change with `--dir`). Math is rendered
with KaTeX from a CDN; without network access the worksheet still works, output is
shown in plain text.

Tests: `.venv/bin/python -m pytest`.

## The language

| Write | Meaning |
|---|---|
| `3 m/s^2`, `2 kg m/s^2`, `4.7 kohm` | a number followed by units. Space or `/` between unit names. |
| `x = 5`, `F = m a` | definition, visible to every cell below |
| `f(x, y) = x y^2` | function definition |
| `expr -> km/hr` | convert the result to other units (`-> deg` for angles) |
| `a == b` | an equation, for `solve`, `nsolve`, `dsolve` |
| `x^2`, `2 pi`, `sin x`, `n!` | powers, implicit multiplication, function application, factorial |
| `[1, 2, 3]`, `matrix([[1, 2], [3, 4]])` | lists and matrices |
| several lines in one cell | evaluated in order, each shown |

If you define a name that is also a unit (`m`, `s`, `F`, `C`, `N`, `g` …), your
definition wins everywhere except directly after a number: `3 m` is always three
meters, `2*m` is twice your `m`. A number that is a divisor or an exponent does not
start a unit phrase, so in `1/2 g t^2` the `g` is still your gravity. The cell shows a
note when a definition shadows a unit.

Trig functions accept degrees or radians (`sin(30 deg)`). Inverse trig returns radians;
add `-> deg`.

The **Reference** panel lists every function, unit and constant with an example that
inserts into the current cell. **Examples** opens complete worksheets.

### Plot cells

Type one or more expressions separated by commas (`sin(x), cos(x)` or `y(t)`), a range
(`0 s` to `5 tau` is fine, units and definitions allowed) and optionally the variable;
it is inferred when there is only one unknown. All series must share units. Axes are
labelled with the SI unit.

## Modules

A module is a folder in `modules/` containing `module.py` with a `register(api)`
function. It adds functions, constants and units to the worksheet namespace and
describes them for the reference panel. Arguments arrive as SymPy objects.

```python
# modules/finance/module.py
import sympy as sp
NAME = "finance"
DESCRIPTION = "Time value of money."

def pmt(rate, n, pv):
    return pv * rate / (1 - (1 + rate) ** -n)

def register(api):
    api.function("pmt", pmt, signature="pmt(rate, n, pv)",
                 doc="level payment for a loan", category="Finance",
                 example="pmt(0.05/12, 360, 300000)")
    api.constant("basis_points", sp.Rational(1, 10000), doc="1 bp")
```

Press ↻ in the reference panel to reload modules without restarting. A module that
fails to import is listed with its error rather than taking the app down. Two modules
ship as examples: `stats` (mean, stdev, linfit, correlation) and `ode` (`dsolve` for
exact solutions with initial values, `odesolve` for numeric solutions that behave like
functions and can be plotted). Extra module folders: `--modules path`.

## How it works

```
quire/engine/parser.py     text → SymPy. Definition detection, == → Eq, -> conversion,
                           unit aliasing after numbers, a restricted eval namespace.
quire/engine/units.py      unit table, dimension checks, conversion, unit stripping.
quire/engine/evaluator.py  top-down evaluation of a document; LaTeX + numeric output.
quire/engine/plotting.py   samples expressions to points (server does no drawing).
quire/engine/worker.py     evaluation in a child process with a timeout and auto-restart.
quire/modules/registry.py  loads modules, builds the namespace and the catalog.
quire/modules/core.py      the built-in module: 190 units, constants and functions.
quire/server.py            stdlib HTTP server: static UI + JSON API.
quire/ui/                  the worksheet UI: vanilla JS, SVG plots, KaTeX rendering.
```

The whole document is re-evaluated on every change. At worksheet scale that takes
milliseconds and keeps the reactive semantics trivially correct. A math cell's result
carries `defines` and `uses`, shown under the cell.

## Known limits

- Evaluation runs in a worker process with a 20 s budget. A cell that exceeds it is
  reported as stopped, the cells below it are skipped for that round, and the worker
  is restarted. Exact `solve` on equations mixing decimals, units and exponentials is
  the usual culprit; `nsolve` answers those in milliseconds.
- Definitions bind at parse time: `F = m a` written *above* `m = 2 kg` stays symbolic,
  which is the intended worksheet reading order.
- A `->` conversion on a function definition is applied when the function is called
  with values. Until then the body is shown in the units it was written in.
- Plot cells draw curves of one variable only. No parametric or 3D plots yet.
- File dialogs are minimal (a name, not a file picker); the on-disk format is plain JSON.
