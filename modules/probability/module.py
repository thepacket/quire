"""Probability: named distributions kept symbolic (sympy.stats), Bayes, expectations of expressions.

    X = normal(mu, sigma)          pdf(X, x)   cdf(X, x)   expected(X^2)   prob(X > 1)
    quantile_of(normal(0, 1), 0.975)   sample_from(poisson(3), 10)
"""
import sympy as sp
from sympy import stats as st

from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "probability"
DESCRIPTION = "Symbolic distributions: pdf, cdf, moments, quantiles, probabilities, Bayes."

_counter = [0]


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _name(prefix):
    _counter[0] += 1
    return f"{prefix}{_counter[0]}"


def _dist(maker, prefix):
    def make(*args):
        try:
            return maker(_name(prefix), *[sp.sympify(a) for a in args])
        except Exception as exc:
            raise EvalError(f"Invalid parameters for {prefix}: {str(exc).splitlines()[0]}") from None
    return make


normal = _dist(st.Normal, "N")
uniform = _dist(st.Uniform, "U")
exponential = _dist(st.Exponential, "Exp")
gamma_dist = _dist(st.Gamma, "Gam")
beta_dist = _dist(st.Beta, "Beta")
lognormal = _dist(st.LogNormal, "LogN")
weibull = _dist(st.Weibull, "Wei")
student_t = _dist(st.StudentT, "T")
chi_squared = _dist(st.ChiSquared, "Chi2")
cauchy = _dist(st.Cauchy, "Cau")
triangular = _dist(st.Triangular, "Tri")
laplace_dist = _dist(st.Laplace, "Lap")
binomial_dist = _dist(st.Binomial, "Bin")
poisson = _dist(st.Poisson, "Poi")
geometric_dist = _dist(st.Geometric, "Geo")
bernoulli_dist = _dist(st.Bernoulli, "Ber")
hypergeometric = _dist(st.Hypergeometric, "Hyp")
negative_binomial = _dist(st.NegativeBinomial, "NB")
discrete_uniform = lambda *vals: st.DiscreteUniform(_name("DU"), [sp.sympify(v) for v in (vals[0] if len(vals) == 1 and isinstance(vals[0], (list, tuple)) else vals)])  # noqa: E731


def _rv(X):
    if not isinstance(X, sp.Basic) or not X.atoms(st.rv.RandomSymbol):
        raise EvalError("Expected a distribution such as normal(mu, sigma), or an expression in one.")
    return X


def pdf(X, x):
    """Density (or mass) function as an expression in x."""
    return sp.simplify(st.density(_rv(X))(sp.sympify(x)))


def cdf(X, x):
    return sp.simplify(st.cdf(_rv(X))(sp.sympify(x)))


def expected(expr):
    """E[expr] for an expression in random variables."""
    return sp.simplify(st.E(_rv(expr)))


def variance_of(expr):
    return sp.simplify(st.variance(_rv(expr)))


def std_of(expr):
    return sp.simplify(st.std(_rv(expr)))


def moment_of(X, n):
    return sp.simplify(st.moment(_rv(X), int(n)))


def skewness_of(X):
    return sp.simplify(st.skewness(_rv(X)))


def kurtosis_of(X):
    return sp.simplify(st.kurtosis(_rv(X)))


def median_of(X):
    m = st.median(_rv(X))
    if isinstance(m, sp.Intersection):
        finite = [a for a in m.args if isinstance(a, sp.FiniteSet)]
        if finite and len(finite[0]) == 1:
            return next(iter(finite[0]))
    if isinstance(m, sp.FiniteSet) and len(m) == 1:
        return next(iter(m))
    return m


def quantile_of(X, p):
    """Value x with P(X <= x) = p."""
    return sp.simplify(st.quantile(_rv(X))(sp.sympify(p)))


def prob(condition):
    """Probability of a condition such as X > 1 or (X > 0) & (X < 2)."""
    if not isinstance(condition, (sp.Rel, sp.And, sp.Or)):
        raise EvalError("prob needs a condition such as X > 1.")
    return sp.simplify(st.P(condition))


def conditional_expected(expr, condition):
    return sp.simplify(st.E(_rv(expr), condition))


def conditional_prob(condition, given):
    return sp.simplify(st.P(condition, given))


def mgf(X, t):
    return sp.simplify(st.moment_generating_function(_rv(X))(sp.sympify(t)))


def entropy_of(X):
    return sp.simplify(st.entropy(_rv(X)))


def sample_from(X, n=10, seed=1):
    import numpy as np

    n = int(n)
    vals = list(st.sample(_rv(X), size=(n,), seed=int(seed)))
    _note(f"{n} pseudo-random samples (seed {int(seed)})")
    return [sp.Float(float(v)) if not float(v).is_integer() else sp.Integer(int(v)) for v in np.asarray(vals).ravel()]


def bayes(prior, likelihood, evidence):
    """P(H|E) = P(E|H) P(H)/P(E)"""
    return sp.simplify(sp.sympify(likelihood) * sp.sympify(prior) / sp.sympify(evidence))


def bayes_update(priors, likelihoods):
    """Posterior probabilities over hypotheses from priors and likelihoods of the observed evidence."""
    pr = [sp.sympify(p) for p in priors]
    lk = [sp.sympify(l) for l in likelihoods]
    if len(pr) != len(lk):
        raise EvalError("priors and likelihoods must have the same length.")
    total = sum(p * l for p, l in zip(pr, lk))
    return [sp.simplify(p * l / total) for p, l in zip(pr, lk)]


def z_score(x, mu, sigma):
    return sp.simplify((sp.sympify(x) - sp.sympify(mu)) / sp.sympify(sigma))


def register(api):
    D = "Probability: distributions"
    for name, fn, sig, doc in [
        ("normal", normal, "normal(mu, sigma)", "normal distribution"),
        ("uniform", uniform, "uniform(a, b)", "continuous uniform on [a, b]"),
        ("exponential", exponential, "exponential(rate)", "exponential distribution"),
        ("gamma_dist", gamma_dist, "gamma_dist(k, theta)", "gamma distribution (shape, scale)"),
        ("beta_dist", beta_dist, "beta_dist(alpha, beta)", "beta distribution"),
        ("lognormal", lognormal, "lognormal(mu, sigma)", "log-normal distribution"),
        ("weibull", weibull, "weibull(alpha, beta)", "Weibull (scale, shape)"),
        ("student_t", student_t, "student_t(nu)", "Student's t"),
        ("chi_squared", chi_squared, "chi_squared(k)", "chi-squared"),
        ("cauchy", cauchy, "cauchy(x0, gamma)", "Cauchy distribution"),
        ("triangular", triangular, "triangular(a, b, c)", "triangular distribution"),
        ("laplace_dist", laplace_dist, "laplace_dist(mu, b)", "Laplace distribution"),
        ("binomial_dist", binomial_dist, "binomial_dist(n, p)", "binomial distribution"),
        ("poisson", poisson, "poisson(lambda)", "Poisson distribution"),
        ("geometric_dist", geometric_dist, "geometric_dist(p)", "geometric distribution"),
        ("bernoulli_dist", bernoulli_dist, "bernoulli_dist(p)", "Bernoulli distribution"),
        ("hypergeometric", hypergeometric, "hypergeometric(N, m, n)", "hypergeometric distribution"),
        ("negative_binomial", negative_binomial, "negative_binomial(r, p)", "negative binomial"),
        ("discrete_uniform", discrete_uniform, "discrete_uniform([values])", "uniform over listed values"),
    ]:
        api.function(name, fn, signature=sig, doc=doc, category=D, example=sig if name in ("normal", "poisson", "binomial_dist") else "")
    Q = "Probability: quantities"
    api.function("pdf", pdf, signature="pdf(X, x)", doc="density or mass function in x", category=Q, example="pdf(normal(mu, sigma), x)")
    api.function("pmf", pdf, signature="pmf(X, k)", doc="probability mass function", category=Q)
    api.function("cdf", cdf, signature="cdf(X, x)", doc="cumulative distribution function", category=Q, example="cdf(normal(0, 1), 1.96)")
    api.function("expected", expected, signature="expected(expr)", doc="E[expr] of an expression in random variables", category=Q,
                 example="expected(normal(mu, sigma)^2)")
    api.function("variance_of", variance_of, signature="variance_of(expr)", doc="Var[expr]", category=Q)
    api.function("std_of", std_of, signature="std_of(expr)", doc="standard deviation", category=Q)
    api.function("moment_of", moment_of, signature="moment_of(X, n)", doc="n-th raw moment", category=Q)
    api.function("skewness_of", skewness_of, signature="skewness_of(X)", doc="skewness", category=Q)
    api.function("kurtosis_of", kurtosis_of, signature="kurtosis_of(X)", doc="kurtosis", category=Q)
    api.function("median_of", median_of, signature="median_of(X)", doc="median", category=Q)
    api.function("quantile_of", quantile_of, signature="quantile_of(X, p)", doc="inverse CDF", category=Q, example="quantile_of(normal(0, 1), 0.975)")
    api.function("prob", prob, signature="prob(X > a)", doc="probability of a condition", category=Q, example="prob(normal(0, 1) > 1.96)")
    api.function("conditional_prob", conditional_prob, signature="conditional_prob(cond, given)", doc="P(cond | given)", category=Q)
    api.function("conditional_expected", conditional_expected, signature="conditional_expected(expr, cond)", doc="E[expr | cond]", category=Q)
    api.function("mgf", mgf, signature="mgf(X, t)", doc="moment generating function", category=Q)
    api.function("entropy_of", entropy_of, signature="entropy_of(X)", doc="differential/Shannon entropy", category=Q)
    api.function("sample_from", sample_from, signature="sample_from(X, n, seed)", doc="pseudo-random samples", category=Q)
    api.function("z_score", z_score, signature="z_score(x, mu, sigma)", doc="(x - μ)/σ", category=Q)
    B = "Probability: Bayes"
    api.function("bayes", bayes, signature="bayes(prior, likelihood, evidence)", doc="P(H|E) = P(E|H) P(H) / P(E)", category=B,
                 example="bayes(0.01, 0.95, 0.01 0.95 + 0.99 0.05)")
    api.function("bayes_update", bayes_update, signature="bayes_update([priors], [likelihoods])", doc="posteriors over hypotheses", category=B,
                 example="bayes_update([1/2, 1/2], [0.8, 0.3])")
