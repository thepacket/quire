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

Tests: `.venv/bin/python -m pytest`. Symbolic coverage report: `.venv/bin/python -m bench.run`
(747 textbook problems across algebra, calculus, equations, linear algebra, transforms,
number theory, special functions, worksheets with units and the stats module, all passing
through the worksheet pipeline).

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

Single-letter unit symbols (`m`, `s`, `g`, `N`, `V`, `C`, `F`, `T` …) are units only in
unit position: directly after a number (`3 m/s^2`) or as a `->` target. Anywhere else
they are ordinary variables, so `F = m a` and `laplace(f, t, s)` mean what they say.
Longer names (`kg`, `Hz`, `meter`, `second`) are always units. A number that is a
divisor or an exponent does not start a unit phrase, so in `1/2 g t^2` the `g` is your
gravity even right after the 2. Physical constants have explicit names: `c_light`,
`G_grav`, `g_0`, `h_planck`, `k_B`, `N_A`, `R_gas`, `e_charge`, `epsilon_0`, `mu_0`.

Assumptions sharpen simplification: `assume x > 0`, `assume n positive integer`,
`assume x, y real`. Prime notation works on defined functions (`f'(2)`, `f''(x)`),
sequences are written `a[n]` (`rsolve(a[n+1] == 2 a[n], a, n, 1)`), and `integral` /
`derivative` are the unevaluated forms of `integrate` / `diff` (`doit` evaluates them).
`implicit_diff(x^2 + y^2 == 1, y, x)` differentiates implicitly; `factor(x^2 - 2, sqrt(2))`
factors over an extension.

A definite integral with numeric bounds that has no closed form is evaluated to 50
digits and matched against a small basis of constants (π, ln 2, Catalan, ζ(3), γ, …) with
an integer-relation search (PSLQ). A match such as `π ln 2 / 8` is returned with a note
saying it was recognized numerically, evidence rather than proof; otherwise a decimal is
shown. `recognize(0.6931471805599453)` does the same for any number, and its coefficient
bound shrinks with the digits you typed so a short decimal cannot be "recognized" by
coincidence. Integrals over (0, ∞) of `g(x)/(e^x ∓ 1)` are done by series expansion and
termwise integration, which is how `Γ(s) ζ(s)` comes out. SymPy's own symbolic attempt
gets an 8 s budget before backends and numerics take over.

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
fails to import is listed with its error rather than taking the app down. Three modules
ship: `stats` (mean, stdev, linfit, correlation), `ode` (`dsolve` for exact solutions
with initial values, `odesolve` for numeric solutions that behave like functions and can
be plotted), and two backend CAS bridges, `maxima` and `fricas`. Extra module folders:
`--modules path`.

### Numerical methods (`modules/numerics`)

Organised by how the approximation is built, and every result carries a note saying so:

| family | functions |
|---|---|
| series and expansions | `taylor`, `chebyshev_approx`, `pade`, `series_solve` |
| iterative | `newton_raphson`, `fixed_point`, `bisection`, `secant`, `jacobi_iter`, `gauss_seidel_iter`, and `*_steps` tables |
| discretized | `finite_difference` (symbolic stencil or number), `fdm_solve`, `bvp_solve`, `fem_solve`, `heat_fdm` |
| time-stepping | `euler`, `heun`, `rk4` with fixed step and `*_steps` tables; `odesolve` in `ode` is the adaptive one |

Solvers return functions of the independent variable, so `y_e = euler(-2 y, y, x, 0, 1, 3, 0.25)`
can be called (`y_e(1)`), plotted against `rk4` and the exact `exp(-2 x)`, and its domain is
enforced. `heat_fdm` returns `u(x, t)`; plot `u(x, 0.05)`. The "Numerical methods" example
worksheet walks through all four families.

### Backends

A module can also register a *fallback* for an operation:

```python
api.fallback("integrate", my_integrate, priority=50)   # also: limit, sum, simplify
```

Backends are tried in priority order (Maxima 10, FriCAS 30).

Core functions call the backends only when SymPy gives up (an unevaluated integral,
limit or sum) or returns something worse than the backend (Meijer G terms, `exp_polar`,
floor-based antiderivatives). Backend simplifications are accepted only if they agree
with the original numerically at random points, because a backend's algebra may assume
principal branches. Backend antiderivatives are accepted only if their derivative matches
the integrand numerically, and definite results with numeric bounds only if they agree
with quadrature.

- `maxima` uses a local Maxima (`brew install maxima`). One short-lived process per call
  with stdin closed; "is it an integer?" questions are answered generically for
  parameters not declared integer, signs are never guessed. Explicit: `maxima_integrate`,
  `maxima_limit`, `maxima_sum`, `maxima_simplify`.
- `fricas` uses a local FriCAS (`brew install fricas`), whose Risch implementation closes
  antiderivatives the others cannot, such as `exp(x) (1 + x)/x^2`. Explicit:
  `fricas_integrate`, `fricas_limit`.

Without the binaries the bridges register nothing and the worksheet works as before.

`python -m bench.run --hard` runs the hard corpus: definite integrals from the tables,
special-function identities and large simplifications. Known gaps are listed in
`bench/hard.py` so a fix shows up as an unexpected pass.

## How it works

```
quire/engine/parser.py     text → SymPy. Definition detection, == → Eq, -> conversion,
                           unit aliasing after numbers, a restricted eval namespace.
quire/engine/units.py      unit table, dimension checks, conversion, unit stripping.
quire/engine/evaluator.py  top-down evaluation of a document; LaTeX + numeric output.
quire/engine/plotting.py   samples expressions to points (server does no drawing).
quire/engine/worker.py     evaluation in a child process with a timeout and auto-restart.
quire/modules/registry.py  loads modules, builds the namespace and the catalog.
quire/modules/core.py      the built-in module, assembled from quire/modules/builtin/*
                           (basics, algebra, calculus, linalg, numbers, special, transforms):
                           about 300 units, constants and functions.
bench/                     the symbolic coverage corpus (bench/problems.py) and its runner;
                           `python -m bench.run` prints pass rates by domain. Every problem
                           also runs under pytest.
quire/server.py            stdlib HTTP server: static UI + JSON API.
quire/ui/                  the worksheet UI: vanilla JS, SVG plots, KaTeX rendering.
```

The whole document is re-evaluated on every change. At worksheet scale that takes
milliseconds and keeps the reactive semantics trivially correct. A math cell's result
carries `defines` and `uses`, shown under the cell.

## Known limits

- Symbolic depth is SymPy plus Maxima plus FriCAS. The hard corpus shows what is still out
  of reach for all three (a Gamma-zeta integral, a Bessel Wronskian, one Meijer-G
  integral, an arctangent addition identity).
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
