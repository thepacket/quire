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

## Deploying on fly.io

The repository carries a `Dockerfile` (Python 3.13 with Maxima and FriCAS from Debian)
and a `fly.toml`. Worksheets persist on a volume mounted at `/data`, and the machine
stops when idle.

```bash
fly launch --copy-config --no-deploy   # creates the app; adjust app name and region in fly.toml
fly volumes create quire_data --size 1 --region yyz
fly secrets set QUIRE_PASSWORD='choose-a-long-password'
fly deploy
```

Set `QUIRE_PASSWORD`: the server evaluates expressions for anyone who can reach it, so
a public URL without a password is an open compute endpoint. With it, the browser asks
once (HTTP Basic auth; any user name, that password). `/health` stays open for the
platform's checks. Locally nothing changes: `python -m quire` still binds to localhost
without a password. The image is about 1 GB because of the two CAS backends; drop the
`apt-get install` line to run on SymPy alone.

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
| `digits 10` | show ten significant digits in every cell below (default 6); `N(x, 20)` and `round(x, 3)` act on one value |
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

Pick a kind, then fill in the fields:

| kind | fields |
|---|---|
| `y = f(x)` | one or more expressions separated by commas (`sin(x), cos(x)` or `y(t)`), a range, optional variable |
| parametric | `x(t)`, `y(t)`, parameter range |
| polar | `r(θ)`, angle range |
| scatter | x data and y data, any expressions that give lists |
| slope field | `dy/dx = f(x, y)`, x and y ranges, density |
| implicit | an equation `F(x, y) == c`, x and y ranges |

Ranges accept units and definitions (`0 s` to `5 tau`); the variable is inferred when
there is only one unknown; all series must share units and axes are labelled with the SI
unit. Function plots are sampled adaptively and broken at discontinuities, so `tan x` has
gaps rather than walls and the visible range ignores the spikes. Hover for values, scroll
to zoom, drag to pan, double-click to reset, click a legend entry to hide a series, tick
`log x` / `log y` for logarithmic axes, and `SVG` downloads the picture.

`a = slider(1, 0, 5, 0.1)` defines a value with a slider under the cell; dragging it
re-evaluates everything below, plots included. The "Interactive plots" example shows
each of these.

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

### Engineering modules

| module | what it carries |
|---|---|
| `circuits` | impedances and phasors, dividers, RC/RL/RLC transfer functions in `s`, cutoff and resonance, step responses, AC power, dB, `bode_gain` / `bode_phase` as expressions in `f` for log plots |
| `control` | `tf`, `tf_poles`, `tf_zeros`, `feedback`, `pid`, `second_order`, `damping`, `overshoot`, `settling_time`, `step_response`, `routh` tables, `is_stable`, `root_locus` and `nyquist` data for scatter plots, state space |
| `mechanics` | section properties, stress and strain, principal and von Mises stresses, beam formulas returning `[deflection(x), moment(x), shear(x)]`, Euler buckling, springs and vibration, Goodman and Soderberg fatigue |
| `thermo` | real fluid properties from CoolProp (`fluid_density(water, T, P)`, saturation, latent heat, psychrometrics), ideal gas and cycle efficiencies, conduction / convection / radiation, LMTD and NTU, Reynolds, friction factor, head loss, Bernoulli, drag |
| `signals` | `sample_signal`, `spectrum` (FFT), windows, convolution, Butterworth and FIR design, `filter_apply`, `freq_response`, `z_transfer`, symbolic Fourier coefficients and partial sums |
| `chemistry` | molar masses and composition from formulas written as names (`molar_mass(CaCO3)`), `balance([C3H8, O2], [CO2, H2O])`, moles and mass, pH, buffers, Arrhenius, Gibbs, Nernst, and a small table of typical material properties (`material_E(steel)`) |

Temperatures are absolute: write `from_celsius(25)` for 25 °C and `to_celsius(T)` to read one.
Rates per second are written `0.1 Hz` or `0.1/second`, because `0.1/s` divides by the
variable `s`. The "Engineering modules" example has one worked problem per module.

### Data and probability modules

| module | what it carries |
|---|---|
| `probability` | distributions that stay symbolic: `normal(mu, sigma)`, `uniform`, `exponential`, `gamma_dist`, `beta_dist`, `lognormal`, `weibull`, `student_t`, `chi_squared`, `binomial_dist`, `poisson`, `geometric_dist`, `bernoulli_dist`, and more; `pdf`, `cdf`, `expected(expr)` for any expression in random variables, `variance_of`, `moment_of`, `quantile_of`, `prob(X > a)`, conditional probability and expectation, `mgf`, `sample_from`; `bayes` and `bayes_update` |
| `stats` | `mean`, `median`, `variance`, `stdev`, `covariance`, `correlation`, `percentile`, `describe`, `histogram`; `linfit`, `linear_regression`, `polyfit`, `expfit`, `powerfit`, `r_squared`; `confidence_interval`, one- and two-sample and paired `t_test`, `chi2_test` (goodness of fit or independence), `anova`, `normality_test`, `bootstrap_mean` |
| `finance` | `fv`, `pv`, `pmt`, `nper`, `rate_solve`, annuities and perpetuities, `amortization` tables, `npv`, `irr`, `payback_period`, effective and real rates, `cagr`, `bond_price`, `bond_duration`, `bond_ytm`, `black_scholes_call` / `put` (symbolic in every parameter), delta and vega |
| `actuarial` | `makeham_qx` mortality rates, `life_table`, `survival`, `life_expectancy`, `annuity_due_factor`, whole-life, term and endowment insurance, `net_annual_premium`, `reserve` |

Rates are per period as decimals; money is a plain number. Every test returns `[statistic, p]`
with a note naming the test. The "Data and probability" example covers all four.

### Science modules

| module | what it carries |
|---|---|
| `physics` | projectile and pendulum kinematics; special relativity (`lorentz_factor`, `time_dilation`, `velocity_addition`, relativistic Doppler); optics (`thin_lens_image`, `snell`, `critical_angle`, `brewster_angle`, gratings, Rayleigh limit); electromagnetism (Coulomb, fields of wires and solenoids, Lorentz force, cyclotron frequency); waves and photons (`photon_energy`, `de_broglie`, `wien_peak`, blackbody); nuclear decay (`decay`, `age_from_fraction`, `binding_energy`) |
| `astronomy` | units and constants (`AU`, `ly`, `pc`, `M_sun`, `M_earth`, `R_earth`, `arcsec`); orbits (`orbital_velocity`, `orbital_period`, `escape_velocity`, `vis_viva`, `hohmann`, `schwarzschild_radius`); magnitudes and distances (`distance_modulus`, `parallax_distance`, `luminosity`, `redshift_velocity`); `julian_date`, `sidereal_time`, `alt_az`, `angular_separation`, `equatorial_to_galactic` |
| `geodesy` | `haversine` and `vincenty` distances, `bearing`, `destination`, `midpoint`, WGS84 `geodetic_to_ecef` / `ecef_to_geodetic`, `utm` |
| `quantum` (information) | Kraus channels (`bit_flip`, `phase_flip`, `depolarizing`, `amplitude_damping`, `phase_damping`, `kraus_apply`, `kraus_valid`), `fidelity_mixed`, `trace_distance`, `bloch_mixed`, `concurrence`, `qft`, `grover_iterate`, `teleport_check` |

Angles are degrees unless written with units; `-> deg` and `-> rad` convert results.
Constants such as `c_light` may be `->` targets, so `velocity_addition(0.5 c_light, 0.5 c_light) -> c_light`
gives `0.8`. The "Science modules" example has a worked problem for each.

### Quantum computing (`modules/quantum`)

Symbolic linear algebra over complex Hilbert spaces, in Dirac notation:

- **States**: `ket(0, 1)`, `qubit(alpha, beta)`, `bloch_state(θ, φ)`, `plus()`, `bell_state(k)`, `ghz(n)`;
  `norm_sq`, `normalize`, `same_state` and `global_phase` (global-phase equivalence), `bloch`,
  `bloch_vector`, `vec` / `to_dirac` to switch between Dirac form and column vectors.
- **Composition**: `tensor`, `is_entangled` and `schmidt` across any split, `density`,
  `partial_trace`, `purity`, `entropy` (von Neumann, bits), `fidelity`.
- **Gates**: `X Y Z H S T CNOT CZ SWAP TOFFOLI`, `Rx Ry Rz phase U3`, `controlled(G)`,
  `gate_on(G, n, qubits...)`, `apply(G, state, qubits...)`, `circuit(n, [[H, 0], [CNOT, 0, 1]])`.
  Qubit 0 is the leftmost symbol in `|q0 q1 ...>`.
- **Observables**: `dagger`, `is_unitary`, `is_hermitian`, `commutator`, `anticommutator`,
  `expectation`, `qvariance`, `uncertainty(A, B, state)` returning `[σ_A σ_B, |<[A,B]>|/2]`.
- **Measurement**: `measure(state, qubits...)` (Born rule table), `born`, `collapse` (projective,
  renormalized), `sample` (simulated shots).

Amplitudes may be symbols: `measure(qubit(alpha, beta))` gives `|α|²` and `|β|²`. The
"Quantum computing" example walks through states, entanglement, unitaries, measurement and
interference. Because `beta`, `gamma`, `zeta` are also function names and `lambda` is a
Python keyword, the parser treats them as variables unless followed by `(`.

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
- Plot cells are 2D: functions, parametric, polar, scatter, slope fields and implicit
  curves. No contour, heatmap or 3D plots yet.
- File dialogs are minimal (a name, not a file picker); the on-disk format is plain JSON.
