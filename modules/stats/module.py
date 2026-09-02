"""Descriptive statistics, regression, intervals and hypothesis tests.

Lists are worksheet lists: mean([1, 2, 3]). Tests return [statistic, p-value] with a note.
"""
import numpy as np
import sympy as sp
from scipy import stats as sps

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "stats"
DESCRIPTION = "Descriptive statistics, regression and fitting, confidence intervals, hypothesis tests."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _values(xs):
    if isinstance(xs, sp.MatrixBase):
        return list(xs)
    if isinstance(xs, (list, tuple)):
        return list(xs)
    raise EvalError("Expected a list, e.g. [1, 2, 3].")


def _floats(xs, what="data"):
    try:
        return np.array([float(U.strip_units(sp.sympify(v))[0]) for v in _values(xs)], dtype=float)
    except (TypeError, ValueError):
        raise EvalError(f"{what} must contain numbers only.") from None


def _f(x):
    return sp.Float(float(x))


# ---- descriptive (symbolic where the data allow)
def mean(xs):
    v = _values(xs)
    return sp.Add(*v) / len(v)


def median(xs):
    v = sorted(_values(xs), key=lambda t: float(t))
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def variance(xs):
    v = _values(xs)
    m = mean(v)
    return sp.Add(*[(t - m) ** 2 for t in v]) / (len(v) - 1)


def stdev(xs):
    return sp.sqrt(variance(xs))


def covariance(xs, ys):
    x, y = _values(xs), _values(ys)
    mx, my = mean(x), mean(y)
    return sp.Add(*[(a - mx) * (b - my) for a, b in zip(x, y)]) / (len(x) - 1)


def correlation(xs, ys):
    x, y = _values(xs), _values(ys)
    mx, my = mean(x), mean(y)
    num = sp.Add(*[(a - mx) * (b - my) for a, b in zip(x, y)])
    den = sp.sqrt(sp.Add(*[(a - mx) ** 2 for a in x]) * sp.Add(*[(b - my) ** 2 for b in y]))
    return num / den


def percentile(xs, q):
    return _f(np.percentile(_floats(xs), float(q)))


def describe(xs):
    """Rows: n, mean, stdev, min, Q1, median, Q3, max."""
    v = _floats(xs)
    rows = [("n", len(v)), ("mean", v.mean()), ("stdev", v.std(ddof=1) if len(v) > 1 else 0.0), ("min", v.min()),
            ("Q1", np.percentile(v, 25)), ("median", np.median(v)), ("Q3", np.percentile(v, 75)), ("max", v.max())]
    return sp.ImmutableMatrix([[sp.Symbol(k), _f(val)] for k, val in rows])


def histogram(xs, bins=10):
    """[bin centres, counts] for a scatter plot."""
    counts, edges = np.histogram(_floats(xs), bins=int(bins))
    centres = (edges[:-1] + edges[1:]) / 2
    return [[_f(c) for c in centres], [sp.Integer(int(c)) for c in counts]]


def zscores(xs):
    v = _floats(xs)
    return [_f(z) for z in (v - v.mean()) / v.std(ddof=1)]


# ---- fitting
def linfit(xs, ys):
    """Least-squares line through the points; returns [slope, intercept]."""
    x, y = _values(xs), _values(ys)
    if len(x) != len(y):
        raise EvalError("x and y lists must have the same length.")
    n = len(x)
    sx, sy = sp.Add(*x), sp.Add(*y)
    sxx = sp.Add(*[t * t for t in x])
    sxy = sp.Add(*[a * b for a, b in zip(x, y)])
    slope = sp.simplify((n * sxy - sx * sy) / (n * sxx - sx ** 2))
    intercept = sp.simplify((sy - slope * sx) / n)
    return [slope, intercept]


def linear_regression(xs, ys):
    """Rows: slope, intercept, r_squared, slope_stderr, p_value."""
    r = sps.linregress(_floats(xs), _floats(ys))
    _note("ordinary least squares; p-value tests slope = 0")
    return sp.ImmutableMatrix([[sp.Symbol("slope"), _f(r.slope)], [sp.Symbol("intercept"), _f(r.intercept)],
                               [sp.Symbol("r_squared"), _f(r.rvalue ** 2)], [sp.Symbol("slope_stderr"), _f(r.stderr)],
                               [sp.Symbol("p_value"), _f(r.pvalue)]])


def polyfit(xs, ys, x, degree=2):
    """Least-squares polynomial of the given degree as an expression in x."""
    if not isinstance(x, sp.Symbol):
        raise EvalError("Give the variable, e.g. polyfit(xs, ys, x, 2).")
    coeffs = np.polyfit(_floats(xs), _floats(ys), int(degree))
    scale = np.max(np.abs(coeffs)) or 1.0
    coeffs = [0.0 if abs(c) < 1e-10 * scale else c for c in coeffs]  # drop rounding noise
    return sum(sp.Float(c, 10) * x ** (len(coeffs) - 1 - k) for k, c in enumerate(coeffs) if c)


def expfit(xs, ys, x):
    """Fit y = a exp(b x) by least squares on ln y."""
    xv, yv = _floats(xs), _floats(ys)
    if np.any(yv <= 0):
        raise EvalError("expfit needs positive y values.")
    b, ln_a = np.polyfit(xv, np.log(yv), 1)
    return sp.Float(np.exp(ln_a), 10) * sp.exp(sp.Float(b, 10) * x)


def powerfit(xs, ys, x):
    """Fit y = a x^b by least squares on logs."""
    xv, yv = _floats(xs), _floats(ys)
    if np.any(yv <= 0) or np.any(xv <= 0):
        raise EvalError("powerfit needs positive x and y values.")
    b, ln_a = np.polyfit(np.log(xv), np.log(yv), 1)
    return sp.Float(np.exp(ln_a), 10) * x ** sp.Float(b, 10)


def r_squared(xs, ys, model, x):
    """Coefficient of determination of a model expression in x against the data."""
    xv, yv = _floats(xs), _floats(ys)
    f = sp.lambdify(x, model, modules="numpy")
    pred = np.asarray(f(xv), dtype=float)
    ss_res = np.sum((yv - pred) ** 2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    return _f(1 - ss_res / ss_tot)


# ---- intervals and tests
def confidence_interval(xs, level=0.95):
    """[low, high] for the mean (t interval)."""
    v = _floats(xs)
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n)
    t = sps.t.ppf((1 + float(level)) / 2, n - 1)
    _note(f"t interval for the mean, {n - 1} degrees of freedom, level {float(level):g}")
    return [_f(v.mean() - t * se), _f(v.mean() + t * se)]


def t_test(xs, mu0=0):
    """One-sample t test: [t, p] against the mean mu0."""
    t, p = sps.ttest_1samp(_floats(xs), float(mu0))
    _note("one-sample t test, two-sided")
    return [_f(t), _f(p)]


def t_test_two(xs, ys):
    """Welch two-sample t test: [t, p]."""
    t, p = sps.ttest_ind(_floats(xs), _floats(ys), equal_var=False)
    _note("Welch t test, two-sided, unequal variances")
    return [_f(t), _f(p)]


def paired_t_test(xs, ys):
    t, p = sps.ttest_rel(_floats(xs), _floats(ys))
    _note("paired t test, two-sided")
    return [_f(t), _f(p)]


def chi2_test(observed, expected=None):
    """Chi-squared goodness of fit [chi2, p]; for a table (list of rows) a test of independence."""
    obs = _values(observed)
    if obs and isinstance(obs[0], (list, tuple, sp.MatrixBase)):
        table = np.array([[float(v) for v in _values(r)] for r in obs])
        chi2, p, dof, _ = sps.chi2_contingency(table)
        _note(f"chi-squared test of independence, {dof} degrees of freedom")
        return [_f(chi2), _f(p)]
    o = _floats(observed)
    e = _floats(expected, "expected") if expected is not None else None
    chi2, p = sps.chisquare(o, e)
    _note("chi-squared goodness-of-fit test")
    return [_f(chi2), _f(p)]


def anova(groups):
    """One-way ANOVA over a list of groups: [F, p]."""
    gs = [_floats(g) for g in _values(groups)]
    F, p = sps.f_oneway(*gs)
    _note(f"one-way ANOVA, {len(gs)} groups")
    return [_f(F), _f(p)]


def normality_test(xs):
    """Shapiro-Wilk [W, p]."""
    w, p = sps.shapiro(_floats(xs))
    return [_f(w), _f(p)]


def bootstrap_mean(xs, n=2000, level=0.95, seed=1):
    """[low, high] percentile bootstrap interval for the mean."""
    v = _floats(xs)
    rng = np.random.default_rng(int(seed))
    means = np.array([rng.choice(v, size=v.size, replace=True).mean() for _ in range(int(n))])
    a = (1 - float(level)) / 2
    _note(f"percentile bootstrap, {int(n)} resamples, seed {int(seed)}")
    return [_f(np.percentile(means, 100 * a)), _f(np.percentile(means, 100 * (1 - a)))]


def register(api):
    S = "Statistics: descriptive"
    api.function("mean", mean, signature="mean([x1, x2, ...])", doc="arithmetic mean", category=S, example="mean([1, 2, 3, 4])")
    api.function("median", median, signature="median([...])", doc="middle value", category=S)
    api.function("variance", variance, signature="variance([...])", doc="sample variance", category=S)
    api.function("stdev", stdev, signature="stdev([...])", doc="sample standard deviation", category=S, example="stdev([2, 4, 4, 4, 5, 5, 7, 9])")
    api.function("covariance", covariance, signature="covariance([x...], [y...])", doc="sample covariance", category=S)
    api.function("correlation", correlation, signature="correlation([x...], [y...])", doc="Pearson correlation", category=S)
    api.function("percentile", percentile, signature="percentile([...], q)", doc="q-th percentile", category=S)
    api.function("describe", describe, signature="describe([...])", doc="n, mean, stdev, quartiles, extremes", category=S)
    api.function("histogram", histogram, signature="histogram([...], bins)", doc="[centres, counts]; plot as scatter", category=S)
    api.function("zscores", zscores, signature="zscores([...])", doc="standardised values", category=S)
    F = "Statistics: fitting"
    api.function("linfit", linfit, signature="linfit([x...], [y...])", doc="least-squares line [slope, intercept]", category=F,
                 example="linfit([0, 1, 2, 3], [1, 3, 5, 7.2])")
    api.function("linear_regression", linear_regression, signature="linear_regression([x...], [y...])", doc="slope, intercept, r², stderr, p",
                 category=F)
    api.function("polyfit", polyfit, signature="polyfit([x...], [y...], x, degree)", doc="least-squares polynomial in x", category=F,
                 example="polyfit([0, 1, 2, 3], [1, 2, 5, 10], x, 2)")
    api.function("expfit", expfit, signature="expfit([x...], [y...], x)", doc="fit a exp(b x)", category=F)
    api.function("powerfit", powerfit, signature="powerfit([x...], [y...], x)", doc="fit a x^b", category=F)
    api.function("r_squared", r_squared, signature="r_squared([x...], [y...], model, x)", doc="R² of a model", category=F)
    T = "Statistics: intervals & tests"
    api.function("confidence_interval", confidence_interval, signature="confidence_interval([...], level)", doc="t interval for the mean",
                 category=T, example="confidence_interval([5.1, 4.9, 5.3, 5.0, 5.2], 0.95)")
    api.function("t_test", t_test, signature="t_test([...], mu0)", doc="one-sample t test [t, p]", category=T)
    api.function("t_test_two", t_test_two, signature="t_test_two([x...], [y...])", doc="Welch two-sample t test [t, p]", category=T)
    api.function("paired_t_test", paired_t_test, signature="paired_t_test([x...], [y...])", doc="paired t test [t, p]", category=T)
    api.function("chi2_test", chi2_test, signature="chi2_test(observed, expected)", doc="goodness of fit or independence [chi2, p]", category=T,
                 example="chi2_test([[10, 20], [30, 40]])")
    api.function("anova", anova, signature="anova([[group1], [group2], ...])", doc="one-way ANOVA [F, p]", category=T)
    api.function("normality_test", normality_test, signature="normality_test([...])", doc="Shapiro-Wilk [W, p]", category=T)
    api.function("bootstrap_mean", bootstrap_mean, signature="bootstrap_mean([...], n, level, seed)", doc="bootstrap interval for the mean",
                 category=T)
