"""Hard corpus: the problems where SymPy is expected to struggle.

Same format as problems.py. This set measures the residue a backend CAS
(Maxima, FriCAS) would have to cover; it is reported separately and failures
here are tracked, not treated as regressions.
"""

HARD = []


def p(domain, cells, expected):
    HARD.append((domain, cells if isinstance(cells, list) else [cells], expected))


# ---------------------------------------------------------------- hard definite integrals
I = "hard integrals"
p(I, "integrate(x/(exp(x) - 1), x, 0, oo)", "pi^2/6")
p(I, "integrate(x^3/(exp(x) - 1), x, 0, oo)", "pi^4/15")
p(I, ["assume s > 1", "integrate(x^(s - 1)/(exp(x) - 1), x, 0, oo)"], "gamma(s) zeta(s)")
p(I, "integrate(ln(x) ln(1 - x), x, 0, 1)", "2 - pi^2/6")
p(I, ["assume a > 0", "integrate(ln(x)/(x^2 + a^2), x, 0, oo)"], "pi ln(a)/(2 a)")
p(I, "integrate(exp(-x^2) ln(x), x, 0, oo)", "-sqrt(pi) (EulerGamma + 2 ln(2))/4")
p(I, "integrate(ln(sin(x)), x, 0, pi/2)", "-pi ln(2)/2")
p(I, ["assume a > 0, b > 0", "integrate(x^a (1 - x)^b, x, 0, 1)"], "beta(a + 1, b + 1)")
p(I, ["assume a > 0, b > 0", "integrate(cos(a x)/(x^2 + b^2), x, 0, oo)"], "pi exp(-a b)/(2 b)")
p(I, "integrate(x exp(-x) sin(x), x, 0, oo)", "1/2")
p(I, "integrate(atan(x)/x, x, 0, 1)", "Catalan")
p(I, "integrate(sin(x^2), x, 0, oo)", "sqrt(2 pi)/4")
p(I, ["assume n > 1", "integrate(1/(1 + x^n), x, 0, oo)"], "pi/(n sin(pi/n))")
p(I, "integrate(exp(-x) ln(x), x, 0, oo)", "-EulerGamma")
p(I, ["assume a > 0, b > 0", "integrate((exp(-a x) - exp(-b x))/x, x, 0, oo)"], "ln(b/a)")
p(I, "integrate(sin(x)^2/x^2, x, 0, oo)", "pi/2")
p(I, "integrate(x^2/(x^4 + 1), x, 0, oo)", "sqrt(2) pi/4")
p(I, "integrate(1/(2 + cos(t)), t, 0, 2 pi)", "2 pi/sqrt(3)")
p(I, ["assume a > 0, b > 0", "integrate((x^a - x^b)/ln(x), x, 0, 1)"], "ln((a + 1)/(b + 1))")
p(I, "integrate(sin(x)/(x (x^2 + 1)), x, 0, oo)", "pi (1 - exp(-1))/2")
p(I, "integrate(ln(1 + x^2)/(1 + x^2), x, 0, oo)", "pi ln(2)")
p(I, "integrate(ln(1 + x)/(1 + x^2), x, 0, 1)", "pi ln(2)/8")
p(I, "integrate(1/((x^2 + 1) (x^2 + 4)), x, 0, oo)", "pi/12")
p(I, "integrate(x/sqrt(1 - x^4), x, 0, 1)", "pi/4")
p(I, "integrate(1/sqrt(1 - x^4), x, 0, 1)", "gamma(1/4)^2/(4 sqrt(2 pi))")
p(I, "integrate(exp(-x) cos(x)^2, x, 0, oo)", "3/5")
p(I, "integrate(ln(x)^2/(1 + x^2), x, 0, oo)", "pi^3/8")
p(I, "integrate(x/(1 + x^3), x, 0, oo)", "2 sqrt(3) pi/9")
p(I, "integrate(exp(-x^2 - 1/x^2), x, 0, oo)", "sqrt(pi) exp(-2)/2")
p(I, "integrate(1/cosh(x), x, -oo, oo)", "pi")
p(I, "integrate(x/sinh(x), x, 0, oo)", "pi^2/4")
p(I, "integrate(ln(x)/(1 - x), x, 0, 1)", "-pi^2/6")
p(I, "integrate(sqrt(x)/(1 + x^2), x, 0, oo)", "sqrt(2) pi/2")
p(I, "integrate(cos(x)/(1 + x^2), x, -oo, oo)", "pi/e")
p(I, "integrate(x^2 exp(-x)/(1 + exp(-x))^2, x, -oo, oo)", "pi^2/3")

# ---------------------------------------------------------------- hard indefinite integrals (checked by differentiation)
J = "hard antiderivatives"
p(J, "simplify(diff(integrate(1/(x^4 + 1), x), x) - 1/(x^4 + 1))", "0")
p(J, "simplify(diff(integrate(1/(sin(x) + cos(x)), x), x) - 1/(sin(x) + cos(x)))", "0")
p(J, "simplify(diff(integrate(sqrt(1 + sqrt(x)), x), x) - sqrt(1 + sqrt(x)))", "0")
p(J, "simplify(diff(integrate(ln(x)/(1 + x), x), x) - ln(x)/(1 + x))", "0")
p(J, "simplify(diff(integrate(x^2/sqrt(1 - x^2), x), x) - x^2/sqrt(1 - x^2))", "0")
p(J, "simplify(diff(integrate(exp(x) (1 + x)/(x^2), x), x) - exp(x) (1 + x)/x^2)", "0")
p(J, ["assume x > 1", "simplify(diff(integrate(1/(x sqrt(x^2 - 1)), x), x) - 1/(x sqrt(x^2 - 1)))"], "0")
p(J, "simplify(diff(integrate(sin(x)^3 cos(x)^2, x), x) - sin(x)^3 cos(x)^2)", "0")
p(J, "simplify(diff(integrate(x atan(x), x), x) - x atan(x))", "0")
p(J, "simplify(diff(integrate(exp(a x) cos(b x), x), x) - exp(a x) cos(b x))", "0")
p(J, "simplify(diff(integrate(1/(1 + exp(x)), x), x) - 1/(1 + exp(x)))", "0")
p(J, ["assume a > 0, a < 1, x > 2", "simplify(diff(integrate(sqrt(x^2 - a^2)/x, x), x) - sqrt(x^2 - a^2)/x)"], "0")
p(J, "simplify(diff(integrate(1/(x^3 - 1), x), x) - 1/(x^3 - 1))", "0")
p(J, "simplify(diff(integrate(x exp(x) sin(x), x), x) - x exp(x) sin(x))", "0")
p(J, "simplify(diff(integrate(1/(2 + cos(x)), x), x) - 1/(2 + cos(x)))", "0")
p(J, "simplify(diff(integrate(ln(x)^2, x), x) - ln(x)^2)", "0")
p(J, "simplify(diff(integrate(sqrt(tan(x)), x), x) - sqrt(tan(x)))", "0")
p(J, "simplify(diff(integrate(1/(x^2 sqrt(x^2 + 1)), x), x) - 1/(x^2 sqrt(x^2 + 1)))", "0")
p(J, "integrate(sin(x)/x, x)", "Si(x)")
p(J, "integrate(exp(x^2), x)", "sqrt(pi) erfi(x)/2")
p(J, "integrate(1/ln(x), x)", "li(x)")

# ---------------------------------------------------------------- special-function identities
F = "special-function identities"
p(F, "simplify(gamma(x) gamma(1 - x) - pi/sin(pi x))", "0")
p(F, "simplify(gamma(x) gamma(x + 1/2) - 2^(1 - 2 x) sqrt(pi) gamma(2 x))", "0")
p(F, "simplify(besselj(n - 1, x) + besselj(n + 1, x) - 2 n besselj(n, x)/x)", "0")
p(F, "simplify(chebyshevt(5, cos(t)) - cos(5 t))", "0")
p(F, "legendre(n, 1)", "1")
p(F, "zeta(6)", "pi^6/945")
p(F, "simplify(beta(a, b) - gamma(a) gamma(b)/gamma(a + b))", "0")
p(F, "polylog(2, 1/2)", "pi^2/12 - ln(2)^2/2")
p(F, "simplify(lambertw(x) exp(lambertw(x)))", "x")
p(F, ["assume n positive integer", "simplify(digamma(n + 1) - harmonic(n) + EulerGamma)"], "0")
p(F, "limit(gamma(x + 1)/(sqrt(2 pi x) (x/e)^x), x, oo)", "1")
p(F, "elliptic_k(1/2)", "gamma(1/4)^2/(4 sqrt(pi))")
p(F, "simplify(erf(x)^2 + erfc(x)^2 + 2 erf(x) erfc(x))", "1")
p(F, "simplify(sinh(asinh(x)) - x)", "0")
p(F, "simplify(besselj(n, x) diff(bessely(n, x), x) - bessely(n, x) diff(besselj(n, x), x))", "2/(pi x)")
p(F, ["assume n integer", "simplify(hermite(n, -x) + (-1)^(n + 1) hermite(n, x))"], "0")
p(F, "simplify(legendre(2, x) - (3 x^2 - 1)/2)", "0")
p(F, "sum(1/k, k, 1, n)", "harmonic(n)")
p(F, "simplify(gamma(x + 1) - x gamma(x))", "0")
p(F, "simplify(expand_func(gamma(x + 3)/gamma(x)))", "x (x + 1) (x + 2)")
p(F, ["assume x > 0", "simplify(atan(x) + atan(1/x))"], "pi/2")
p(F, "simplify(asin(x) + acos(x))", "pi/2")
p(F, "series(besselj(0, x), x, 0, 6)", "1 - x^2/4 + x^4/64")
p(F, "diff(zeta(s), s)", {"ok": True})
p(F, "simplify(erf(x) - 2/sqrt(pi) integrate(exp(-t^2), t, 0, x))", "0")

# ---------------------------------------------------------------- large-expression simplification
G = "large simplification"
p(G, "simplify(expand((a + b)^6)/expand((a + b)^4))", "(a + b)^2")
p(G, "simplify(sin(x)^6 + cos(x)^6 + 3 sin(x)^2 cos(x)^2)", "1")
p(G, "simplify((sqrt(x^2 + 1) - x) (sqrt(x^2 + 1) + x))", "1")
p(G, "simplify((1 + tan(x)^2) cos(x)^2)", "1")
p(G, ["assume x > 0", "simplify(ln(x^2 + 2 x + 1) - 2 ln(x + 1))"], "0")
p(G, "sqrtdenest(sqrt(2 + sqrt(3)))", "sqrt(6)/2 + sqrt(2)/2")
p(G, ["assume x > 0, y > 0", "simplify(exp(ln(x) + ln(y)))"], "x y")
p(G, "simplify((a x^2 + b x + c) - a (x + b/(2 a))^2 - (c - b^2/(4 a)))", "0")
p(G, "factor(x^12 - 1)", {"contains": "x**4 - x**2 + 1"})
p(G, "factor(expand((x + y + z)^4))", "(x + y + z)^4")
p(G, "simplify(det(matrix([[a, b, c, d], [b, a, d, c], [c, d, a, b], [d, c, b, a]])))", "(a + b + c + d) (a - b + c - d) (a + b - c - d) (a - b - c + d)")
p(G, "simplify(cos(x)^4 - sin(x)^4 - cos(2 x))", "0")
p(G, "simplify((exp(i x) + exp(-i x))/2 - cos(x))", "0")
p(G, "simplify(tan(x/2) - sin(x)/(1 + cos(x)))", "0")
p(G, ["assume x > 1", "simplify(sqrt(x + 2 sqrt(x - 1)) - sqrt(x - 1) - 1)"], "0")
p(G, "simplify(sinh(x)^2 - cosh(x)^2 + 1)", "0")
p(G, "simplify((x^6 - 1)/(x^3 - 1) - x^3 - 1)", "0")
p(G, "simplify(sum(sin(k x), k, 1, n) - sin(n x/2) sin((n + 1) x/2)/sin(x/2))", "0")
p(G, "simplify(expand((x + 1)^20) - (x + 1)^20)", "0")
p(G, "simplify(cancel((x^10 - 1)/(x - 1)) - sum(x^k, k, 0, 9))", "0")
p(G, "simplify((a + b)^3 - a^3 - b^3 - 3 a b (a + b))", "0")
p(G, "simplify(1/(1 + 1/(1 + 1/(1 + 1/x))))", "(2 x + 1)/(3 x + 2)")
p(G, ["assume x real", "simplify(sqrt(x^2 + 2 x + 1) - abs(x + 1))"], "0")
p(G, "trigsimp(sin(x)^2 cos(y)^2 + cos(x)^2 sin(y)^2 + 2 sin(x) cos(x) sin(y) cos(y))", "sin(x + y)^2")
p(G, "simplify(log(exp(x)) - x)", {"ok": True})
p(G, "simplify((sqrt(5) + 1)/2 - 1/((sqrt(5) - 1)/2))", "0")
p(G, "simplify(acos(x) - atan(sqrt(1 - x^2)/x))", {"ok": True})
p(G, "simplify(2 cos(x)^2 - 1 - cos(2 x))", "0")
p(G, "simplify(cos(3 x) - 4 cos(x)^3 + 3 cos(x))", "0")
p(G, ["assume n positive integer", "simplify((-1)^n (-1)^n)"], "1")


# Known gaps: neither SymPy nor the Maxima backend closes these yet. Tracked, not counted as regressions.
KNOWN_GAPS = {
    "assume s > 1 | integrate(x^(s - 1)/(exp(x) - 1), x, 0, oo)": "Gamma(s) zeta(s): neither SymPy nor Maxima",
    "integrate(ln(1 + x^2)/(1 + x^2), x, 0, oo)": "SymPy returns a Meijer G term; Maxima leaves it unevaluated",
    "integrate(ln(1 + x)/(1 + x^2), x, 0, 1)": "SymPy hangs; Maxima gives a complex polylog form",
    "simplify(diff(integrate(exp(x) (1 + x)/(x^2), x), x) - exp(x) (1 + x)/x^2)": "SymPy's simplify hangs on expint forms",
    "simplify(besselj(n, x) diff(bessely(n, x), x) - bessely(n, x) diff(besselj(n, x), x))": "Bessel Wronskian not known to SymPy",
    "assume x > 0 | simplify(atan(x) + atan(1/x))": "no arctangent addition rule in SymPy or Maxima",
}
