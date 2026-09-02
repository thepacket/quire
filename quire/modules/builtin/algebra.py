"""Simplification, polynomials, substitution, equations and inequalities."""
import sympy as sp

from ...engine import units as U
from ...engine.errors import EvalError
from .. import hooks
from ._util import as_list, as_residual, pretty_solutions, prune_piecewise, sym

_CANDIDATE_UNITS = None


def _candidate_units():
    global _CANDIDATE_UNITS
    if _CANDIDATE_UNITS is None:
        u = U.u
        _CANDIDATE_UNITS = [sp.S.One, u.m, u.s, u.kg, u.A, u.K, u.mol, u.cd, u.m / u.s, u.m / u.s**2, u.N, u.J,
                            u.W, u.Pa, u.V, u.ohm, u.C, u.F, u.henry, u.Hz, u.m**2, u.m**3, u.kg / u.m**3,
                            u.N * u.m, u.J / u.K, u.W / u.m**2, 1 / u.s, u.kg / u.s, u.m**3 / u.s, u.T, u.Wb]
    return _CANDIDATE_UNITS


def _needs_stripping(exprs):
    """True when a transcendental function has units inside its argument (sympy's solve hangs on those)."""
    for e in exprs:
        for f in e.atoms(sp.Function):
            if any(U.has_units(a) for a in f.args):
                return True
    return False


def _infer_unit(residuals, s, value):
    for cand in _candidate_units():
        try:
            for r in residuals:
                bound = r.subs(s, value * cand)
                U.check_dimensions(U.to_base(bound))
                U.to_base(bound)
        except Exception:  # noqa: BLE001
            continue
        return cand
    return None


def _symbols(syms):
    if len(syms) == 1 and isinstance(syms[0], (list, tuple)):
        syms = tuple(syms[0])
    return tuple(sym(s) for s in syms)


def _solve(eqs, *syms):
    syms = _symbols(syms)
    eq_list = list(eqs) if isinstance(eqs, (list, tuple)) else [eqs]
    residuals = [as_residual(e) for e in eq_list]
    if not syms:
        free = set().union(*[r.free_symbols for r in residuals])
        if len(free) != 1:
            raise EvalError("Say which variable to solve for: solve(equation, x).")
        syms = (free.pop(),)
    floats = any(r.atoms(sp.Float) for r in residuals)
    strip = any(U.has_units(r) for r in residuals) and _needs_stripping(residuals)
    work = [U.strip_units(r)[0] for r in residuals] if strip else residuals
    kw = dict(dict=False, rational=False if floats else None)
    try:
        res = sp.solve(work if len(work) > 1 else work[0], *syms, **kw)
    except NotImplementedError:
        # Equations with abs() need real unknowns; retry with real stand-ins.
        reals = {s: sp.Dummy(s.name, real=True) for s in syms if not s.is_real}
        try:
            res = sp.solve([w.subs(reals) for w in work] if len(work) > 1 else work[0].subs(reals),
                           *[reals.get(s, s) for s in syms], **kw)
            back = {v: k for k, v in reals.items()}
            res = sp.sympify(res).subs(back) if not isinstance(res, list) else [sp.sympify(r).subs(back) for r in res]
            res = [tuple(r) if isinstance(r, sp.Tuple) else r for r in res] if isinstance(res, list) else res
        except NotImplementedError:
            names = ", ".join(str(s) for s in syms)
            raise EvalError(f"No exact solution found for {names}. Try nsolve(equation, {names}, guess) "
                            f"for a numeric answer.") from None
    if floats and isinstance(res, list) and res:
        real = [r for r in res if getattr(r, "is_real", None) is not False
                and not (isinstance(r, sp.Expr) and r.has(sp.I))]
        if real:
            res = real
    if strip and len(syms) == 1 and isinstance(res, list) and res:
        unit = _infer_unit(residuals, syms[0], res[0])
        if unit is not None and unit != 1:
            res = [r * unit for r in res]
    return pretty_solutions(res, syms)


def _solve_real(eq, x):
    r = sp.solveset(as_residual(eq), sym(x), domain=sp.S.Reals)
    if r is sp.S.EmptySet:
        return []
    if isinstance(r, sp.FiniteSet):
        items = list(r)
        try:
            return sorted(items, key=lambda v: float(v))
        except (TypeError, ValueError):
            return items
    return r


def _solve_ineq(ineqs, *syms):
    syms = _symbols(syms)
    if isinstance(ineqs, (list, tuple)):
        return sp.reduce_inequalities(list(ineqs), list(syms) or None)
    if len(syms) == 1:
        return sp.solve_univariate_inequality(ineqs, syms[0], relational=True)
    return sp.reduce_inequalities(ineqs, list(syms) or None)


def _nsolve(eqs, syms, guess):
    if isinstance(eqs, (list, tuple)):
        eqs = [U.strip_units(as_residual(e))[0] for e in eqs]
    else:
        eqs = U.strip_units(as_residual(eqs))[0]
    try:
        return sp.nsolve(eqs, syms, guess)
    except Exception as exc:
        raise EvalError(f"nsolve failed: {str(exc).splitlines()[0]}") from None


def _linsolve(eqs, *syms):
    syms = _symbols(syms)
    res = sp.linsolve(list(eqs) if isinstance(eqs, (list, tuple)) else [eqs], *syms)
    return pretty_solutions(res, syms)


def numerically_equal(a, b, tries: int = 4) -> bool:
    """Compare two expressions at random points that respect the symbols' sign assumptions."""
    import random

    syms = sorted((a.free_symbols | b.free_symbols), key=lambda s: s.name)
    if not syms:
        try:
            return abs(complex(sp.N(a - b, 20))) < 1e-9
        except (TypeError, ValueError):
            return False
    rng = random.Random(7)
    ok = 0
    for _ in range(tries * 3):
        point = {}
        for s_ in syms:
            v = sp.Rational(rng.randint(1, 60), rng.randint(1, 9)) + (sp.Rational(1, 7) if not s_.is_integer else 0)
            if s_.is_integer:
                v = sp.Integer(rng.randint(2, 12))
            if s_.is_negative:
                v = -v
            elif not (s_.is_positive or s_.is_nonnegative) and rng.random() < 0.4:
                v = -v
            if s_.is_real is None and not s_.is_integer:
                # No real assumption: sample off the real axis, where branch cuts show up.
                v = v + sp.I * sp.Rational(rng.randint(-40, 40), rng.randint(1, 9))
            point[s_] = v
        try:
            da = complex(sp.N(a.subs(point), 30))
            db = complex(sp.N(b.subs(point), 30))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            continue
        if abs(da - db) > 1e-9 * max(1.0, abs(da), abs(db)):
            return False
        ok += 1
        if ok >= tries:
            return True
    return False


def denest_sqrt(expr):
    """sqrt(A + c sqrt(B)) -> sqrt(p) + sqrt(q) when A^2 - c^2 B is a perfect square (symbolic denesting)."""
    def one(pw):
        base = pw.base
        if not isinstance(base, sp.Add):
            return pw
        for term in base.args:
            c, rad = term.as_coeff_Mul()
            if isinstance(rad, sp.Pow) and rad.exp == sp.S.Half:
                A, B = base - term, rad.base
                D = sp.factor(A ** 2 - c ** 2 * B)
                r = sp.powdenest(sp.sqrt(D), force=True).replace(sp.Abs, lambda z: z)  # p, q are symmetric in r
                if any(pp.exp == sp.S.Half for pp in r.atoms(sp.Pow)):
                    continue
                pq = ((A + r) / 2, (A - r) / 2)
                return sp.sqrt(sp.factor(pq[0])) + sp.sign(c) * sp.sqrt(sp.factor(pq[1]))
        return pw

    return expr.replace(lambda e: isinstance(e, sp.Pow) and e.exp == sp.S.Half and isinstance(e.base, sp.Add), one)


def _candidates(expr):
    """Cheap targeted rewrites that sympy's simplify does not try on its own."""
    if expr.has(sp.erfc, sp.erfi, sp.erf2):
        yield expr.rewrite(sp.erf)
    if expr.has(sp.acos, sp.acot, sp.asec, sp.acsc):
        yield expr.rewrite(sp.asin).rewrite(sp.atan)
    if expr.has(sp.LambertW):
        w = sp.Wild("w")
        yield expr.replace(sp.exp(sp.LambertW(w)), lambda w: w / sp.LambertW(w)) if False else \
            expr.replace(lambda e: isinstance(e, sp.exp) and isinstance(e.args[0], sp.LambertW),
                         lambda e: e.args[0].args[0] / e.args[0])
    if expr.has(sp.polygamma, sp.gamma, sp.binomial, sp.factorial, sp.harmonic):
        yield sp.expand_func(expr)
    if expr.has(sp.polygamma):
        yield expr.rewrite(sp.harmonic)
    if any(isinstance(pw, sp.Pow) and pw.exp == sp.S.Half for pw in expr.atoms(sp.Pow)):
        inner = expr.replace(lambda e: isinstance(e, sp.Pow) and e.exp == sp.S.Half,
                             lambda e: sp.sqrt(sp.factor(e.base)))
        yield sp.refine(inner)
    if any(isinstance(pw, sp.Pow) and pw.exp == sp.S.Half and pw.base.has(sp.Pow) for pw in expr.atoms(sp.Pow)):
        yield sp.sqrtdenest(expr)
        alt = denest_sqrt(expr)
        if alt != expr and numerically_equal(alt, expr):
            yield alt
    if expr.has(sp.sin, sp.cos, sp.tan):
        yield sp.trigsimp(sp.expand_trig(expr))


def _simplify(expr):
    expr = prune_piecewise(sp.sympify(expr), hooks.context.get("bounds", {}))
    res = prune_piecewise(sp.simplify(expr), hooks.context.get("bounds", {}))
    for cand in _candidates(res):
        try:
            alt = sp.simplify(cand)
        except Exception:  # noqa: BLE001
            continue
        if sp.count_ops(alt) < sp.count_ops(res):
            res = alt
    if hooks.available("simplify") and sp.count_ops(res) > 0:
        alt = hooks.run("simplify", res)
        # A backend's algebra may assume principal branches; accept only what checks out numerically.
        if alt is not None and sp.count_ops(alt) < sp.count_ops(res) and numerically_equal(alt, res):
            res = alt
    return res


def _subs(expr, *pairs):
    if len(pairs) == 2:
        return sp.sympify(expr).subs(pairs[0], pairs[1])
    if len(pairs) == 1 and isinstance(pairs[0], (list, tuple)):
        return sp.sympify(expr).subs([tuple(p) for p in pairs[0]])
    raise EvalError("Use subs(expr, x, value) or subs(expr, [[x, 1], [y, 2]]).")


def _rewrite(expr, target):
    return sp.sympify(expr).rewrite(target)


def _coeff(expr, x, n=1):
    return sp.Poly(expr, x).coeff_monomial(x ** n) if n else sp.Poly(expr, x).coeff_monomial(1)


def _coeffs(expr, x):
    return sp.Poly(expr, x).all_coeffs()


def _roots(expr, x=None):
    r = sp.roots(as_residual(expr), x)
    out = []
    for root, mult in r.items():
        out.extend([root] * mult)
    return out


def _interpolate(points, x):
    pts = [tuple(p) for p in as_list(points)]
    return sp.interpolate(pts, sym(x))


def _piecewise(*pieces):
    return sp.Piecewise(*[tuple(p) for p in pieces])


def _N(expr, digits=15):
    return sp.N(expr, int(digits))


def register(api):
    G = "Simplify & expand"
    api.function("simplify", _simplify, signature="simplify(expr)", doc="simplify an expression", category=G)
    api.function("expand", sp.expand, signature="expand(expr)", doc="multiply out", category=G,
                 example="expand((x+1)^3)")
    api.function("factor", lambda e, ext=None: sp.factor(e, extension=ext) if ext is not None else sp.factor(e),
                 signature="factor(expr, ext)", doc="factor a polynomial; ext = sqrt(2) or i to factor over an extension",
                 category=G, example="factor(x^2 - 1)")
    api.function("collect", sp.collect, signature="collect(expr, x)", doc="group by powers of x", category=G)
    api.function("cancel", sp.cancel, signature="cancel(expr)", doc="cancel common factors", category=G)
    api.function("apart", sp.apart, signature="apart(expr, x)", doc="partial fractions", category=G,
                 example="apart(1/(x^2 - 1), x)")
    api.function("together", sp.together, signature="together(expr)", doc="combine over a common denominator",
                 category=G)
    api.function("trigsimp", sp.trigsimp, signature="trigsimp(expr)", doc="simplify trig identities", category=G)
    api.function("expand_trig", sp.expand_trig, signature="expand_trig(expr)", doc="expand sin(2x) etc.",
                 category=G, example="expand_trig(sin(2 x))")
    api.function("expand_log", lambda e: sp.expand_log(e, force=False), signature="expand_log(expr)",
                 doc="split logarithms (needs positive symbols)", category=G, example="expand_log(ln(x y))")
    api.function("logcombine", sp.logcombine, signature="logcombine(expr)", doc="combine logarithms", category=G)
    api.function("powsimp", sp.powsimp, signature="powsimp(expr)", doc="combine powers", category=G,
                 example="powsimp(x^a x^b)")
    api.function("radsimp", sp.radsimp, signature="radsimp(expr)", doc="rationalise denominators", category=G,
                 example="radsimp(1/(1 + sqrt(2)))")
    api.function("sqrtdenest", sp.sqrtdenest, signature="sqrtdenest(expr)", doc="denest nested square roots",
                 category=G, example="sqrtdenest(sqrt(5 + 2 sqrt(6)))")
    api.function("ratsimp", sp.ratsimp, signature="ratsimp(expr)", doc="simplify a rational function", category=G)
    api.function("gammasimp", sp.gammasimp, signature="gammasimp(expr)", doc="simplify gamma/factorial ratios",
                 category=G)
    api.function("separatevars", sp.separatevars, signature="separatevars(expr)",
                 doc="factor into functions of single variables", category=G)
    api.function("rewrite", _rewrite, signature="rewrite(expr, target)", doc="rewrite in terms of target function",
                 category=G, example="rewrite(cos(x), exp)")
    api.function("nsimplify", sp.nsimplify, signature="nsimplify(x)", doc="find an exact form for a decimal",
                 category=G, example="nsimplify(0.5)")
    api.function("subs", _subs, signature="subs(expr, x, value)", doc="substitute a value", category=G,
                 example="subs(x^2, x, 3)")
    api.function("N", _N, signature="N(expr, digits)", doc="numeric value", category=G, example="N(pi, 20)")
    api.function("numer", sp.numer, signature="numer(expr)", doc="numerator", category=G)
    api.function("denom", sp.denom, signature="denom(expr)", doc="denominator", category=G)
    api.function("piecewise", _piecewise, signature="piecewise((expr, cond), ...)",
                 doc="piecewise definition", category=G, example="piecewise((x, x < 1), (1, x >= 1))")
    api.function("Eq", sp.Eq, signature="a == b", doc="an equation (write a == b)", category=G)

    P = "Polynomials"
    api.function("degree", lambda e, x: sp.degree(e, x), signature="degree(p, x)", doc="degree in x", category=P)
    api.function("coeff", _coeff, signature="coeff(p, x, n)", doc="coefficient of x^n", category=P,
                 example="coeff(3 x^2 + 2 x, x, 2)")
    api.function("coeffs", _coeffs, signature="coeffs(p, x)", doc="all coefficients, highest first", category=P)
    api.function("roots", _roots, signature="roots(p, x)", doc="polynomial roots with multiplicity", category=P,
                 example="roots(x^3 - 3 x^2 + 3 x - 1, x)")
    api.function("discriminant", sp.discriminant, signature="discriminant(p, x)", doc="discriminant", category=P,
                 example="discriminant(a x^2 + b x + c, x)")
    api.function("resultant", sp.resultant, signature="resultant(p, q, x)", doc="resultant", category=P)
    api.function("groebner", lambda ps, *xs: list(sp.groebner(list(ps), *xs)), signature="groebner([p, q], x, y)",
                 doc="Gröbner basis", category=P)
    api.function("quo", sp.quo, signature="quo(p, q, x)", doc="polynomial quotient", category=P)
    api.function("rem", sp.rem, signature="rem(p, q, x)", doc="polynomial remainder", category=P)
    api.function("horner", sp.horner, signature="horner(p, x)", doc="Horner form", category=P)
    api.function("interpolate", _interpolate, signature="interpolate([[x1, y1], ...], x)",
                 doc="polynomial through the points", category=P, example="interpolate([[1, 1], [2, 4], [3, 9]], x)")

    E = "Equations"
    api.function("solve", _solve, signature="solve(eq, x)", doc="solve exactly; eq is 'a == b' or a list",
                 category=E, example="solve(x^2 == 4, x)")
    api.function("solve_real", _solve_real, signature="solve_real(eq, x)", doc="real solutions only, as a set",
                 category=E, example="solve_real(x^4 == 1, x)")
    api.function("solve_ineq", _solve_ineq, signature="solve_ineq(ineq, x)", doc="solve an inequality",
                 category=E, example="solve_ineq(x^2 - 4 < 0, x)")
    api.function("linsolve", _linsolve, signature="linsolve([eqs], x, y)", doc="solve a linear system",
                 category=E, example="linsolve([x + y == 3, x - y == 1], x, y)")
    api.function("nsolve", _nsolve, signature="nsolve(eq, x, guess)", doc="solve numerically from a guess",
                 category=E, example="nsolve(cos(x) == x, x, 1)")
