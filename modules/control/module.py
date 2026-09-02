"""Control systems: transfer functions in s, poles and zeros, responses, stability, PID, state space."""
import sympy as sp

from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "control"
DESCRIPTION = "Transfer functions, poles/zeros, step responses, Routh stability, PID, state space."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _sym(x):
    if not isinstance(x, sp.Symbol):
        raise EvalError(f"Expected a variable name such as s, got '{x}'.")
    return x


def tf(num, den, s):
    """Transfer function from coefficient lists (highest power first) or expressions."""
    s = _sym(s)
    def poly(c):
        if isinstance(c, (list, tuple)):
            return sum(sp.sympify(a) * s ** (len(c) - 1 - k) for k, a in enumerate(c))
        return sp.sympify(c)
    return sp.cancel(poly(num) / poly(den))


def tf_poles(H, s):
    n, d = sp.fraction(sp.cancel(sp.sympify(H)))
    return sp.Poly(d, _sym(s)).all_roots() if sp.Poly(d, s).degree() > 0 else []


def tf_zeros(H, s):
    n, d = sp.fraction(sp.cancel(sp.sympify(H)))
    return sp.Poly(n, _sym(s)).all_roots() if sp.Poly(n, s).degree() > 0 else []


def dc_gain(H, s):
    return sp.limit(sp.sympify(H), _sym(s), 0)


def is_stable(H, s):
    """All poles strictly in the left half-plane (evaluated numerically)."""
    ps = tf_poles(H, s)
    if any(p.free_symbols for p in ps):
        raise EvalError("Stability needs numeric coefficients; use routh for a symbolic table.")
    return all(sp.re(sp.N(p)) < 0 for p in ps)


def routh(den, s):
    """Routh array of a polynomial (rows as a matrix); sign changes in column 1 = unstable poles."""
    s = _sym(s)
    poly = sp.Poly(sp.sympify(den) if not isinstance(den, (list, tuple)) else sum(sp.sympify(a) * s ** (len(den) - 1 - k) for k, a in enumerate(den)), s)
    c = poly.all_coeffs()
    n = len(c)
    cols = (n + 1) // 2
    rows = [[c[i] if i < n else 0 for i in range(0, n, 2)], [c[i] if i < n else 0 for i in range(1, n, 2)]]
    for r in rows:
        r.extend([0] * (cols - len(r)))
    for i in range(2, n):
        prev, prev2 = rows[i - 1], rows[i - 2]
        if prev[0] == 0:
            eps = sp.Symbol("epsilon", positive=True)
            prev[0] = eps
            _note("a zero in the first column was replaced by a small positive epsilon")
        row = []
        for j in range(cols - 1):
            row.append(sp.simplify((prev[0] * prev2[j + 1] - prev2[0] * prev[j + 1]) / prev[0]))
        row.append(0)
        rows.append(row)
    _note("Routh array: the number of sign changes in the first column is the number of right-half-plane poles")
    return sp.ImmutableMatrix(rows)


def feedback(G, H=1, s=None):
    """Closed loop G/(1 + G H) (negative feedback)."""
    return sp.cancel(sp.sympify(G) / (1 + sp.sympify(G) * sp.sympify(H)))


def series_tf(*G):
    return sp.cancel(sp.Mul(*[sp.sympify(g) for g in G]))


def parallel_tf(*G):
    return sp.cancel(sp.Add(*[sp.sympify(g) for g in G]))


def pid(Kp, Ki, Kd, s):
    return sp.sympify(Kp) + sp.sympify(Ki) / _sym(s) + sp.sympify(Kd) * s


def second_order(wn, zeta, s):
    """Standard second-order system wn^2/(s^2 + 2 zeta wn s + wn^2)."""
    return wn ** 2 / (_sym(s) ** 2 + 2 * zeta * wn * s + wn ** 2)


def damping(H, s):
    """[wn, zeta] of a second-order denominator."""
    n, d = sp.fraction(sp.cancel(sp.sympify(H)))
    p = sp.Poly(d, _sym(s))
    if p.degree() != 2:
        raise EvalError("damping needs a second-order transfer function.")
    a2, a1, a0 = p.all_coeffs()
    wn = sp.sqrt(a0 / a2)
    zeta = sp.simplify(a1 / (2 * a2 * wn))
    return [sp.simplify(wn), zeta]


def overshoot(zeta):
    """Percent overshoot of a second-order step response."""
    z = sp.sympify(zeta)
    return 100 * sp.exp(-sp.pi * z / sp.sqrt(1 - z ** 2))


def settling_time(wn, zeta, tolerance=0.02):
    """2 % settling time estimate 4/(zeta wn) (or -ln(tol)/(zeta wn))."""
    return sp.simplify(-sp.log(sp.sympify(tolerance)) / (zeta * wn))


def rise_time(wn, zeta):
    """Approximate 10-90 % rise time (1.8/wn for zeta near 0.5, Franklin's estimate)."""
    return sp.simplify((1 + 1.1 * zeta + 1.4 * zeta ** 2) / wn)


def step_response(H, s, t):
    """Inverse Laplace transform of H(s)/s."""
    y = sp.inverse_laplace_transform(sp.apart(sp.sympify(H) / _sym(s), s), s, t)
    return sp.simplify(y.rewrite(sp.Heaviside).subs(sp.Heaviside(t), 1))


def impulse_response(H, s, t):
    y = sp.inverse_laplace_transform(sp.apart(sp.sympify(H), _sym(s)), s, t)
    return sp.simplify(y.subs(sp.Heaviside(t), 1))


def steady_state_error(G, s, kind=0):
    """Steady-state error to a unit step (kind 0), ramp (1) or parabola (2) with unity feedback."""
    G = sp.sympify(G)
    s = _sym(s)
    k = int(kind)
    err = sp.limit(s * (1 / s ** (k + 1)) / (1 + G), s, 0)
    return sp.simplify(err)


def state_space_tf(A, B, C, D, s):
    """C (sI - A)^-1 B + D"""
    A, B, C, D = (sp.Matrix(m) for m in (A, B, C, D))
    n = A.shape[0]
    return sp.simplify(C * (sp.eye(n) * _sym(s) - A).inv() * B + D)


def controllability(A, B):
    A, B = sp.Matrix(A), sp.Matrix(B)
    n = A.shape[0]
    M = sp.Matrix.hstack(*[A ** k * B for k in range(n)])
    return M.rank() == n


def observability(A, C):
    A, C = sp.Matrix(A), sp.Matrix(C)
    n = A.shape[0]
    M = sp.Matrix.vstack(*[C * A ** k for k in range(n)])
    return M.rank() == n


def root_locus(G, s, k_max=10, n=200):
    """Closed-loop pole locations for gains 0..k_max: [real parts, imaginary parts] for a scatter plot."""
    import numpy as np

    G = sp.sympify(G)
    s = _sym(s)
    num, den = sp.fraction(sp.cancel(G))
    pn, pd = sp.Poly(num, s), sp.Poly(den, s)
    re, im = [], []
    for k in np.linspace(0, float(k_max), int(n)):
        coeffs = (pd + pn * sp.Rational(str(round(k, 6)))).all_coeffs()
        roots = np.roots([float(c) for c in coeffs])
        re.extend(float(r.real) for r in roots)
        im.extend(float(r.imag) for r in roots)
    _note(f"root locus: closed-loop poles of 1 + K G(s) = 0 for K from 0 to {float(k_max):g}")
    return [re, im]


def nyquist(H, s, f_min=0.01, f_max=100, n=400):
    """[real parts, imaginary parts] of H(j2πf) for a scatter/line plot."""
    import numpy as np

    H = sp.sympify(H)
    fn = sp.lambdify(_sym(s), H, modules="numpy")
    fs = np.logspace(np.log10(float(f_min)), np.log10(float(f_max)), int(n))
    vals = np.asarray(fn(2j * np.pi * fs), dtype=complex)
    return [[float(v) for v in vals.real], [float(v) for v in vals.imag]]


def register(api):
    T = "Control: transfer functions"
    api.function("tf", tf, signature="tf(num, den, s)", doc="transfer function from coefficient lists or expressions",
                 category=T, example="tf([1], [1, 2, 1], s)")
    api.function("tf_poles", tf_poles, signature="tf_poles(H, s)", doc="poles", category=T, example="tf_poles(1/(s^2 + 3 s + 2), s)")
    api.function("tf_zeros", tf_zeros, signature="tf_zeros(H, s)", doc="zeros", category=T)
    api.function("dc_gain", dc_gain, signature="dc_gain(H, s)", doc="H(0)", category=T)
    api.function("feedback", feedback, signature="feedback(G, H)", doc="closed loop G/(1 + G H)", category=T,
                 example="feedback(K/(s (s + 1)), 1)")
    api.function("series_tf", series_tf, signature="series_tf(G1, G2)", doc="cascade", category=T)
    api.function("parallel_tf", parallel_tf, signature="parallel_tf(G1, G2)", doc="parallel", category=T)
    api.function("pid", pid, signature="pid(Kp, Ki, Kd, s)", doc="Kp + Ki/s + Kd s", category=T, example="pid(2, 1, 0.5, s)")
    api.function("second_order", second_order, signature="second_order(wn, zeta, s)", doc="standard second-order system",
                 category=T, example="second_order(wn, zeta, s)")
    api.function("state_space_tf", state_space_tf, signature="state_space_tf(A, B, C, D, s)", doc="C (sI - A)^-1 B + D",
                 category=T)
    R = "Control: responses & stability"
    api.function("step_response", step_response, signature="step_response(H, s, t)", doc="unit step response y(t)",
                 category=R, example="step_response(1/(s + 1), s, t)")
    api.function("impulse_response", impulse_response, signature="impulse_response(H, s, t)", doc="impulse response",
                 category=R)
    api.function("is_stable", is_stable, signature="is_stable(H, s)", doc="all poles in the left half-plane?", category=R)
    api.function("routh", routh, signature="routh(den, s)", doc="Routh array of a characteristic polynomial", category=R,
                 example="routh(s^3 + 2 s^2 + 3 s + K, s)")
    api.function("damping", damping, signature="damping(H, s)", doc="[wn, zeta] of a second-order system", category=R)
    api.function("overshoot", overshoot, signature="overshoot(zeta)", doc="percent overshoot", category=R,
                 example="overshoot(0.5)")
    api.function("settling_time", settling_time, signature="settling_time(wn, zeta, tol)", doc="settling time estimate",
                 category=R)
    api.function("rise_time", rise_time, signature="rise_time(wn, zeta)", doc="rise time estimate", category=R)
    api.function("steady_state_error", steady_state_error, signature="steady_state_error(G, s, kind)",
                 doc="error to step (0), ramp (1), parabola (2) with unity feedback", category=R)
    api.function("controllability", controllability, signature="controllability(A, B)", doc="controllable?", category=R)
    api.function("observability", observability, signature="observability(A, C)", doc="observable?", category=R)
    api.function("root_locus", root_locus, signature="root_locus(G, s, k_max, n)",
                 doc="[re, im] of closed-loop poles as gain varies; plot as scatter", category=R,
                 example="root_locus(1/(s (s + 1) (s + 2)), s, 10)")
    api.function("nyquist", nyquist, signature="nyquist(H, s, f_min, f_max)", doc="[re, im] of H(j2πf); plot as scatter",
                 category=R)
