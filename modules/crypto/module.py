"""Number theory and cryptography: continued fractions, Pell and Diophantine equations,
RSA and Diffie-Hellman worked step by step, discrete logarithms, elliptic curves over F_p."""
import math

import sympy as sp
from sympy import ntheory as nt

from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "crypto"
DESCRIPTION = "Continued fractions, Pell equations, RSA, Diffie-Hellman, discrete logs, elliptic curves mod p."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _int(x, what="argument"):
    x = sp.sympify(x)
    if not x.is_Integer:
        raise EvalError(f"{what} must be an integer, got {x}.")
    return int(x)


# ---- continued fractions and Pell
def continued_fraction(x, n=10):
    """Partial quotients [a0; a1, a2, ...] of x (exact for rationals and quadratic surds)."""
    x = sp.sympify(x)
    try:
        cf = sp.continued_fraction(x)
        out = []
        for a in cf:
            if isinstance(a, list):  # periodic part: repeat it up to n terms
                _note(f"periodic part {a} repeats")
                while len(out) < int(n):
                    out.extend(sp.Integer(v) for v in a)
                break
            out.append(sp.Integer(a))
            if len(out) >= int(n):
                break
        return out[: int(n)]
    except Exception:  # noqa: BLE001
        out = []
        v = sp.N(x, 60)
        for _ in range(int(n)):
            a = sp.floor(v)
            out.append(sp.Integer(a))
            if v == a:
                break
            v = 1 / (v - a)
        return out


def convergents(x, n=8):
    cf = continued_fraction(x, n)
    return list(sp.continued_fraction_convergents([int(a) for a in cf]))[: int(n)]


def from_continued_fraction(terms):
    return sp.continued_fraction_reduce([int(t) for t in terms])


def pell(D):
    """Fundamental solution [x, y] of x^2 - D y^2 = 1."""
    from sympy.solvers.diophantine.diophantine import diop_DN

    D = _int(D, "D")
    if int(math.isqrt(D)) ** 2 == D:
        raise EvalError("D must not be a perfect square.")
    sols = diop_DN(D, 1)
    x, y = min(sols, key=lambda s: s[0])
    _note(f"fundamental solution of x^2 - {D} y^2 = 1; further solutions from (x + y sqrt({D}))^k")
    return [sp.Integer(x), sp.Integer(y)]


def diophantine_solve(eq):
    """Integer solutions of a polynomial equation (parametric with t_0, ... when infinite)."""
    from sympy.solvers.diophantine import diophantine

    expr = eq.lhs - eq.rhs if isinstance(eq, sp.Eq) else sp.sympify(eq)
    sols = diophantine(expr)
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    _note("solutions listed as [" + ", ".join(str(s) for s in syms) + "]")
    return [list(s) for s in sols]


def crt(remainders, moduli):
    """Chinese remainder theorem: x with x ≡ r_i (mod m_i); returns [x, M]."""
    r = [int(v) for v in remainders]
    m = [int(v) for v in moduli]
    x, M = nt.modular.crt(m, r)
    return [sp.Integer(x), sp.Integer(M)]


def primitive_root(p):
    return sp.Integer(nt.primitive_root(_int(p, "p")))


def multiplicative_order(a, n):
    return sp.Integer(nt.n_order(_int(a, "a"), _int(n, "n")))


def discrete_log(p, g, h):
    """x with g^x ≡ h (mod p)."""
    return sp.Integer(nt.discrete_log(_int(p, "p"), _int(h, "h"), _int(g, "g")))


def is_quadratic_residue(a, p):
    return nt.is_quad_residue(_int(a, "a"), _int(p, "p"))


def sqrt_mod(a, p):
    r = nt.sqrt_mod(_int(a, "a"), _int(p, "p"))
    if r is None:
        raise EvalError("No square root modulo p.")
    return sp.Integer(r)


# ---- RSA and Diffie-Hellman
def rsa_keygen(p, q, e=65537):
    """[n, e, d] from primes p, q and public exponent e."""
    p, q, e = _int(p, "p"), _int(q, "q"), _int(e, "e")
    if not (sp.isprime(p) and sp.isprime(q)):
        raise EvalError("p and q must be prime.")
    n = p * q
    phi = (p - 1) * (q - 1)
    if math.gcd(e, phi) != 1:
        raise EvalError(f"e must be coprime to phi(n) = {phi}.")
    d = pow(e, -1, phi)
    _note(f"n = {p}·{q} = {n}, phi(n) = {phi}, d = e^-1 mod phi(n) = {d}")
    return [sp.Integer(n), sp.Integer(e), sp.Integer(d)]


def rsa_encrypt(m, e, n):
    m, e, n = _int(m, "m"), _int(e, "e"), _int(n, "n")
    if not 0 <= m < n:
        raise EvalError("The message must satisfy 0 <= m < n.")
    return sp.Integer(pow(m, e, n))


def rsa_decrypt(c, d, n):
    return sp.Integer(pow(_int(c, "c"), _int(d, "d"), _int(n, "n")))


def diffie_hellman(p, g, a, b):
    """[A, B, shared secret] for private keys a and b."""
    p, g, a, b = (_int(v, n) for v, n in ((p, "p"), (g, "g"), (a, "a"), (b, "b")))
    A, B = pow(g, a, p), pow(g, b, p)
    sA, sB = pow(B, a, p), pow(A, b, p)
    _note(f"A = g^a mod p = {A}, B = g^b mod p = {B}; both sides compute {sA}")
    if sA != sB:
        raise EvalError("Shared secrets differ (internal error).")
    return [sp.Integer(A), sp.Integer(B), sp.Integer(sA)]


# ---- elliptic curves y^2 = x^3 + a x + b over F_p
def _on_curve(P, a, b, p):
    if P is None:
        return True
    x, y = P
    return (y * y - (x ** 3 + a * x + b)) % p == 0


def _ec_add(P, Q, a, p):
    if P is None:
        return Q
    if Q is None:
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and (y1 + y2) % p == 0:
        return None
    if P == Q:
        lam = (3 * x1 * x1 + a) * pow(2 * y1, -1, p) % p
    else:
        lam = (y2 - y1) * pow(x2 - x1, -1, p) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)


def _pt(P):
    if P is None or (isinstance(P, sp.Symbol) and P.name == "O"):
        return None
    if isinstance(P, (list, tuple, sp.MatrixBase)) and len(P) == 2:
        return (int(P[0]), int(P[1]))
    raise EvalError("A curve point is [x, y]; O is the point at infinity.")


def _out(P):
    return sp.Symbol("O") if P is None else [sp.Integer(P[0]), sp.Integer(P[1])]


def ec_add(P, Q, a, b, p):
    a, b, p = _int(a, "a"), _int(b, "b"), _int(p, "p")
    P, Q = _pt(P), _pt(Q)
    for R in (P, Q):
        if not _on_curve(R, a, b, p):
            raise EvalError(f"{R} is not on y^2 = x^3 + {a}x + {b} mod {p}.")
    return _out(_ec_add(P, Q, a, p))


def ec_multiply(k, P, a, b, p):
    a, b, p, k = _int(a, "a"), _int(b, "b"), _int(p, "p"), _int(k, "k")
    P = _pt(P)
    if not _on_curve(P, a, b, p):
        raise EvalError("P is not on the curve.")
    R, Q = None, P
    while k > 0:
        if k & 1:
            R = _ec_add(R, Q, a, p)
        Q = _ec_add(Q, Q, a, p)
        k >>= 1
    return _out(R)


def ec_points(a, b, p):
    """All points of y^2 = x^3 + a x + b over F_p (small p), plus O."""
    a, b, p = _int(a, "a"), _int(b, "b"), _int(p, "p")
    if p > 500:
        raise EvalError("ec_points lists points for p <= 500; use ec_order for larger primes.")
    pts = [[sp.Integer(x), sp.Integer(y)] for x in range(p) for y in range(p) if (y * y - x ** 3 - a * x - b) % p == 0]
    _note(f"{len(pts) + 1} points including O")
    return pts


def ec_order(P, a, b, p):
    """Order of the point P (smallest k with kP = O)."""
    a, b, p = _int(a, "a"), _int(b, "b"), _int(p, "p")
    P0 = _pt(P)
    R, k = P0, 1
    while R is not None:
        R = _ec_add(R, P0, a, p)
        k += 1
        if k > 4 * p + 10:
            raise EvalError("Order search exceeded the Hasse bound; check the point.")
    return sp.Integer(k)


def ecdh(P, a, b, p, k_a, k_b):
    """Elliptic-curve Diffie-Hellman: [A, B, shared point]."""
    A = ec_multiply(k_a, P, a, b, p)
    Bp = ec_multiply(k_b, P, a, b, p)
    S1 = ec_multiply(k_a, Bp, a, b, p)
    S2 = ec_multiply(k_b, A, a, b, p)
    if S1 != S2:
        raise EvalError("Shared points differ (internal error).")
    return [A, Bp, S1]


def register(api):
    N = "Crypto: number theory"
    api.function("continued_fraction", continued_fraction, signature="continued_fraction(x, n)", doc="partial quotients", category=N,
                 example="continued_fraction(sqrt(2), 6)")
    api.function("convergents", convergents, signature="convergents(x, n)", doc="rational convergents", category=N, example="convergents(pi, 5)")
    api.function("from_continued_fraction", from_continued_fraction, signature="from_continued_fraction([a0, a1, ...])", doc="value of a finite continued fraction", category=N)
    api.function("pell", pell, signature="pell(D)", doc="fundamental solution of x² - D y² = 1", category=N, example="pell(61)")
    api.function("diophantine_solve", diophantine_solve, signature="diophantine_solve(eq)", doc="integer solutions", category=N,
                 example="diophantine_solve(3 x + 5 y == 1)")
    api.function("crt", crt, signature="crt([r1, r2], [m1, m2])", doc="Chinese remainder theorem [x, M]", category=N, example="crt([2, 3, 2], [3, 5, 7])")
    api.function("primitive_root", primitive_root, signature="primitive_root(p)", doc="smallest primitive root", category=N)
    api.function("multiplicative_order", multiplicative_order, signature="multiplicative_order(a, n)", doc="order of a mod n", category=N)
    api.function("discrete_log", discrete_log, signature="discrete_log(p, g, h)", doc="x with g^x ≡ h (mod p)", category=N, example="discrete_log(23, 5, 8)")
    api.function("is_quadratic_residue", is_quadratic_residue, signature="is_quadratic_residue(a, p)", doc="a a square mod p?", category=N)
    api.function("sqrt_mod", sqrt_mod, signature="sqrt_mod(a, p)", doc="square root modulo p", category=N)
    C = "Crypto: protocols"
    api.function("rsa_keygen", rsa_keygen, signature="rsa_keygen(p, q, e)", doc="[n, e, d]", category=C, example="rsa_keygen(61, 53, 17)")
    api.function("rsa_encrypt", rsa_encrypt, signature="rsa_encrypt(m, e, n)", doc="m^e mod n", category=C, example="rsa_encrypt(65, 17, 3233)")
    api.function("rsa_decrypt", rsa_decrypt, signature="rsa_decrypt(c, d, n)", doc="c^d mod n", category=C)
    api.function("diffie_hellman", diffie_hellman, signature="diffie_hellman(p, g, a, b)", doc="[A, B, shared]", category=C,
                 example="diffie_hellman(23, 5, 6, 15)")
    api.function("ec_add", ec_add, signature="ec_add(P, Q, a, b, p)", doc="point addition on y² = x³ + ax + b mod p", category=C,
                 example="ec_add([5, 1], [5, 1], 2, 2, 17)")
    api.function("ec_multiply", ec_multiply, signature="ec_multiply(k, P, a, b, p)", doc="scalar multiple kP", category=C)
    api.function("ec_points", ec_points, signature="ec_points(a, b, p)", doc="all points for small p", category=C, example="ec_points(2, 2, 17)")
    api.function("ec_order", ec_order, signature="ec_order(P, a, b, p)", doc="order of a point", category=C)
    api.function("ecdh", ecdh, signature="ecdh(P, a, b, p, k_a, k_b)", doc="elliptic-curve Diffie-Hellman", category=C)
