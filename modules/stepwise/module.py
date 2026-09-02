"""Step-by-step derivations: differentiation, integration, partial fractions, Gaussian elimination,
linear and quadratic equations. Each function returns the steps and the result."""
import sympy as sp
from sympy.integrals import manualintegrate as mi

from quire.engine.errors import EvalError
from quire.engine.steps import Steps
from quire.modules import hooks

NAME = "stepwise"
DESCRIPTION = "Derivations shown step by step: derivatives, integrals, partial fractions, elimination, equations."


def _sym(x):
    if not isinstance(x, sp.Symbol):
        raise EvalError(f"Expected a variable name, got '{x}'.")
    return x


# ---------------------------------------------------------------- differentiation
def _d(expr, x, steps):
    """Return the derivative of expr, appending the rule applied."""
    D = lambda e: sp.Derivative(e, x)  # noqa: E731
    if not expr.has(x):
        steps.append((f"constant rule: the derivative of a constant is 0", sp.Eq(D(expr), 0, evaluate=False)))
        return sp.S.Zero
    if expr == x:
        steps.append(("the derivative of x with respect to itself is 1", sp.Eq(D(x), 1, evaluate=False)))
        return sp.S.One
    if isinstance(expr, sp.Add):
        steps.append(("sum rule: differentiate each term", sp.Eq(D(expr), sp.Add(*[D(t) for t in expr.args], evaluate=False), evaluate=False)))
        return sp.Add(*[_d(t, x, steps) for t in expr.args])
    if isinstance(expr, sp.Mul):
        const, rest = expr.as_independent(x)
        if const != 1:
            steps.append((f"constant multiple rule: keep the factor {const}", sp.Eq(D(expr), const * D(rest), evaluate=False)))
            return const * _d(rest, x, steps)
        factors = list(expr.args)
        f, g = factors[0], sp.Mul(*factors[1:])
        steps.append(("product rule: (f g)' = f' g + f g'", sp.Eq(D(expr), D(f) * g + f * D(g), evaluate=False)))
        return _d(f, x, steps) * g + f * _d(g, x, steps)
    if isinstance(expr, sp.Pow):
        base, ex = expr.base, expr.exp
        if not ex.has(x):
            if base == x:
                steps.append((f"power rule: d/dx x^n = n x^(n-1) with n = {ex}", sp.Eq(D(expr), ex * x ** (ex - 1), evaluate=False)))
                return ex * x ** (ex - 1)
            steps.append((f"power rule with the chain rule: n u^(n-1) u' where u = {base}", sp.Eq(D(expr), ex * base ** (ex - 1) * D(base), evaluate=False)))
            return ex * base ** (ex - 1) * _d(base, x, steps)
        if not base.has(x):
            steps.append((f"exponential rule: d/dx a^u = a^u ln(a) u' with a = {base}", sp.Eq(D(expr), expr * sp.log(base) * D(ex), evaluate=False)))
            return expr * sp.log(base) * _d(ex, x, steps)
        steps.append(("logarithmic differentiation: f^g = exp(g ln f)", sp.Eq(D(expr), expr * D(ex * sp.log(base)), evaluate=False)))
        return expr * _d(ex * sp.log(base), x, steps)
    if isinstance(expr, sp.Function) and len(expr.args) == 1:
        u = expr.args[0]
        outer = sp.diff(expr.func(sp.Symbol("u")), sp.Symbol("u"))
        name = expr.func.__name__
        if u == x:
            steps.append((f"table: d/dx {name}(x) = {outer.subs(sp.Symbol('u'), x)}", sp.Eq(D(expr), outer.subs(sp.Symbol("u"), x), evaluate=False)))
            return outer.subs(sp.Symbol("u"), x)
        steps.append((f"chain rule: {name}'(u) u' with u = {u}", sp.Eq(D(expr), outer.subs(sp.Symbol("u"), u) * D(u), evaluate=False)))
        return outer.subs(sp.Symbol("u"), u) * _d(u, x, steps)
    result = sp.diff(expr, x)
    steps.append(("differentiate directly", sp.Eq(D(expr), result, evaluate=False)))
    return result


def steps_diff(f, x):
    x = _sym(x)
    f = sp.sympify(f)
    steps = []
    raw = _d(f, x, steps)
    result = sp.simplify(raw)
    if raw != result:
        steps.append(("collect and simplify", sp.Eq(raw, result, evaluate=False)))
    return Steps(result, steps, f"d/d{x} of {f}")


# ---------------------------------------------------------------- integration (manualintegrate rules)
def _eval_rule(rule):
    try:
        return rule.eval()
    except Exception:  # noqa: BLE001
        return sp.integrate(rule.integrand, rule.variable)


def _describe_rule(rule, x, steps):
    try:
        _describe_rule_inner(rule, x, steps)
    except Exception as exc:  # noqa: BLE001 - never let a description problem hide the result
        steps.append((f"({type(rule).__name__.replace('Rule', '').lower()} rule)", None))


def _describe_rule_inner(rule, x, steps):
    name = type(rule).__name__
    integrand = getattr(rule, "integrand", None)
    I = lambda e: sp.Integral(e, x)  # noqa: E731
    if name == "ConstantRule":
        steps.append(("constant rule", sp.Eq(I(integrand), integrand * x, evaluate=False)))
    elif name == "PowerRule":
        steps.append((f"power rule: x^n -> x^(n+1)/(n+1) with n = {rule.exp}", sp.Eq(I(integrand), rule.base ** (rule.exp + 1) / (rule.exp + 1), evaluate=False)))
    elif name == "AddRule":
        steps.append(("sum rule: integrate each term", sp.Eq(I(integrand), sp.Add(*[I(getattr(r, 'integrand', 0)) for r in rule.substeps], evaluate=False), evaluate=False)))
        for r in rule.substeps:
            _describe_rule(r, x, steps)
    elif name == "ConstantTimesRule":
        steps.append((f"constant multiple: take out {rule.constant}", sp.Eq(I(integrand), rule.constant * I(rule.other), evaluate=False)))
        _describe_rule(rule.substep, x, steps)
    elif name == "URule":
        steps.append((f"substitution u = {rule.u_func}, du = {sp.diff(rule.u_func, x)} d{x}", sp.Eq(I(integrand), sp.Integral(rule.substep.integrand, rule.u_var), evaluate=False)))
        _describe_rule(rule.substep, rule.u_var, steps)
        steps.append((f"substitute back u = {rule.u_func}", None))
    elif name == "PartsRule":
        u, dv = rule.u, rule.dv
        steps.append((f"integration by parts with u = {u}, dv = {dv} d{x}: uv - integral of v du", None))
        _describe_rule(rule.v_step, x, steps)
        if getattr(rule, "second_step", None) is not None:
            _describe_rule(rule.second_step, x, steps)
    elif name in ("ExpRule", "TrigRule", "ArctanRule", "ArcsinRule", "LogRule", "ReciprocalRule", "SinRule", "CosRule",
                  "ArccoshRule", "ArcsinhRule", "Log1pRule"):
        steps.append((f"table integral ({name.replace('Rule', '').lower()})", sp.Eq(I(integrand), _eval_rule(rule), evaluate=False)))
    elif name == "PartialFractionsRule":
        steps.append(("partial fractions", sp.Eq(I(integrand), I(rule.substep.integrand), evaluate=False)))
        _describe_rule(rule.substep, x, steps)
    elif name == "RewriteRule":
        steps.append((f"rewrite the integrand", sp.Eq(I(integrand), I(rule.rewritten), evaluate=False)))
        _describe_rule(rule.substep, x, steps)
    elif name in ("AlternativeRule",):
        _describe_rule(rule.alternatives[0], x, steps)
    elif name == "PiecewiseRule":
        for sub, cond in rule.subfunctions:
            steps.append((f"case {cond}", None))
            _describe_rule(sub, x, steps)
    elif name == "DontKnowRule":
        steps.append(("no elementary rule applies here; sympy's general integrator is used", None))
    else:
        sub = getattr(rule, "substep", None)
        steps.append((f"{name.replace('Rule', '')} rule", None))
        if sub is not None:
            _describe_rule(sub, x, steps)


def steps_integrate(f, x, a=None, b=None):
    x = _sym(x)
    f = sp.sympify(f)
    steps = []
    try:
        rule = mi.integral_steps(f, x)
    except Exception:  # noqa: BLE001
        rule = None
        steps.append(("no rule tree available; using the general integrator", None))
    if rule is not None:
        _describe_rule(rule, x, steps)
    anti = sp.integrate(f, x)
    steps.append(("antiderivative", sp.Eq(sp.Integral(f, x), anti, evaluate=False)))
    if a is not None and b is not None:
        val = sp.simplify(anti.subs(x, b) - anti.subs(x, a))
        steps.append((f"evaluate from {a} to {b}: F({b}) - F({a})", sp.Eq(sp.Integral(f, (x, a, b)), val, evaluate=False)))
        return Steps(val, steps, f"integral of {f} from {a} to {b}")
    return Steps(anti, steps, f"integral of {f}")


# ---------------------------------------------------------------- partial fractions
def steps_partial_fractions(expr, x):
    x = _sym(x)
    expr = sp.together(sp.sympify(expr))
    num, den = sp.fraction(expr)
    steps = []
    fden = sp.factor(den)
    steps.append(("factor the denominator", sp.Eq(den, fden, evaluate=False)))
    result = sp.apart(expr, x)
    terms = sp.Add.make_args(result)
    coeffs = sp.symbols(f"A1:{len(terms) + 1}")
    form = sp.Add(*[c * sp.fraction(t)[1] ** -1 if sp.fraction(t)[1] != 1 else c for c, t in zip(coeffs, terms)])
    steps.append(("write the form with unknown constants", sp.Eq(expr, form, evaluate=False)))
    cleared = sp.expand(sp.together(form) * den) if den != 1 else form
    steps.append(("multiply through by the denominator and match coefficients", sp.Eq(sp.expand(num), sp.expand(sp.cancel(form * den)), evaluate=False)))
    steps.append(("solve for the constants", sp.Eq(form, result, evaluate=False)))
    return Steps(result, steps, "partial fractions")


# ---------------------------------------------------------------- Gaussian elimination
def steps_gauss(A, b=None):
    """Row-reduce [A | b] (or A) to reduced row echelon form, one row operation per step."""
    M = sp.Matrix(A)
    if b is not None:
        M = M.row_join(sp.Matrix(list(b)) if not isinstance(b, sp.MatrixBase) else sp.Matrix(b))
    steps = [("start", sp.ImmutableMatrix(M))]
    rows, cols = M.shape
    r = 0
    for c in range(cols if b is None else cols - 1):
        if r >= rows:
            break
        pivot = next((i for i in range(r, rows) if M[i, c] != 0), None)
        if pivot is None:
            continue
        if pivot != r:
            M.row_swap(pivot, r)
            steps.append((f"swap R{r + 1} and R{pivot + 1}", sp.ImmutableMatrix(M)))
        if M[r, c] != 1:
            factor = M[r, c]
            M[r, :] = M[r, :] / factor
            steps.append((f"R{r + 1} := R{r + 1} / ({factor})", sp.ImmutableMatrix(M)))
        for i in range(rows):
            if i != r and M[i, c] != 0:
                k = M[i, c]
                M[i, :] = M[i, :] - k * M[r, :]
                steps.append((f"R{i + 1} := R{i + 1} - ({k}) R{r + 1}", sp.ImmutableMatrix(M)))
        r += 1
    return Steps(sp.ImmutableMatrix(M), steps, "Gaussian elimination to reduced row echelon form")


# ---------------------------------------------------------------- equations
def steps_solve(eq, x):
    """Linear or quadratic equation in x, solved by isolating or by the quadratic formula."""
    x = _sym(x)
    if not isinstance(eq, sp.Eq):
        raise EvalError("Write the equation with ==, e.g. steps_solve(2 x + 3 == 7, x).")
    steps = []
    expr = sp.expand(eq.lhs - eq.rhs)
    steps.append(("move everything to one side", sp.Eq(expr, 0, evaluate=False)))
    poly = sp.Poly(expr, x)
    deg = poly.degree()
    if deg == 1:
        a, b = poly.all_coeffs()
        steps.append((f"{'add' if b < 0 else 'subtract'} {abs(b)} on both sides", sp.Eq(a * x, -b, evaluate=False)))
        sol = sp.simplify(-b / a)
        steps.append((f"divide by {a}", sp.Eq(x, sol, evaluate=False)))
        return Steps([sol], steps, "solve a linear equation")
    if deg == 2:
        a, b, c = poly.all_coeffs()
        disc = sp.simplify(b ** 2 - 4 * a * c)
        steps.append((f"identify a = {a}, b = {b}, c = {c}", None))
        steps.append(("discriminant b^2 - 4ac", sp.Eq(sp.Symbol("Delta"), disc, evaluate=False)))
        r1, r2 = sp.simplify((-b - sp.sqrt(disc)) / (2 * a)), sp.simplify((-b + sp.sqrt(disc)) / (2 * a))
        steps.append(("quadratic formula x = (-b ± sqrt(Delta)) / (2a)", sp.Eq(x, sp.FiniteSet(r1, r2), evaluate=False)))
        return Steps(sorted({r1, r2}, key=lambda v: str(v)), steps, "solve a quadratic equation")
    raise EvalError("steps_solve handles linear and quadratic equations; use solve for others.")


def register(api):
    S = "Step by step"
    api.function("steps_diff", steps_diff, signature="steps_diff(f, x)", doc="derivative with each rule shown", category=S,
                 example="steps_diff(x^2 sin(x), x)")
    api.function("steps_integrate", steps_integrate, signature="steps_integrate(f, x, a, b)", doc="integral with each rule shown", category=S,
                 example="steps_integrate(x exp(x), x)")
    api.function("steps_partial_fractions", steps_partial_fractions, signature="steps_partial_fractions(expr, x)", doc="partial fraction decomposition",
                 category=S, example="steps_partial_fractions((3 x + 5)/(x^2 + 3 x + 2), x)")
    api.function("steps_gauss", steps_gauss, signature="steps_gauss(A, b)", doc="row reduction, one operation per step", category=S,
                 example="steps_gauss(matrix([[2, 1], [1, 3]]), [3, 5])")
    api.function("steps_solve", steps_solve, signature="steps_solve(eq, x)", doc="linear or quadratic equation, step by step", category=S,
                 example="steps_solve(x^2 - 5 x + 6 == 0, x)")
