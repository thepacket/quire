"""Number theory and combinatorics."""
import sympy as sp
from sympy import ntheory
from sympy.functions.combinatorial import numbers as cn


def _factorint(n):
    return [[p, e] for p, e in sp.factorint(int(n)).items()]


def register(api):
    N = "Number theory"
    api.function("isprime", lambda n: sp.isprime(int(n)), signature="isprime(n)", doc="primality test", category=N,
                 example="isprime(97)")
    api.function("prime", lambda n: sp.prime(int(n)), signature="prime(n)", doc="the n-th prime", category=N)
    api.function("nextprime", lambda n: sp.nextprime(int(n)), signature="nextprime(n)", doc="next prime after n",
                 category=N)
    api.function("prevprime", lambda n: sp.prevprime(int(n)), signature="prevprime(n)", doc="previous prime",
                 category=N)
    api.function("primefactors", lambda n: sp.primefactors(int(n)), signature="primefactors(n)",
                 doc="distinct prime factors", category=N, example="primefactors(360)")
    api.function("factorint", _factorint, signature="factorint(n)", doc="[[prime, exponent], ...]", category=N,
                 example="factorint(360)")
    api.function("divisors", lambda n: sp.divisors(int(n)), signature="divisors(n)", doc="all divisors",
                 category=N)
    api.function("totient", lambda n: sp.totient(int(n)), signature="totient(n)", doc="Euler's φ(n)", category=N)
    api.function("invmod", lambda a, m: sp.mod_inverse(int(a), int(m)), signature="invmod(a, m)",
                 doc="modular inverse of a mod m", category=N, example="invmod(3, 7)")
    api.function("powmod", lambda a, b, m: sp.Integer(pow(int(a), int(b), int(m))), signature="powmod(a, b, m)",
                 doc="a^b mod m", category=N, example="powmod(2, 100, 7)")
    api.function("isqrt", lambda n: sp.integer_nthroot(int(n), 2)[0], signature="isqrt(n)",
                 doc="integer square root", category=N)
    api.function("legendre_symbol", lambda a, p: ntheory.legendre_symbol(int(a), int(p)),
                 signature="legendre_symbol(a, p)", doc="Legendre symbol (a/p)", category=N)

    K = "Combinatorics"
    api.function("fibonacci", cn.fibonacci, signature="fibonacci(n)", doc="Fibonacci number", category=K,
                 example="fibonacci(10)")
    api.function("lucas", cn.lucas, signature="lucas(n)", doc="Lucas number", category=K)
    api.function("bernoulli", cn.bernoulli, signature="bernoulli(n)", doc="Bernoulli number", category=K,
                 example="bernoulli(4)")
    api.function("euler_number", cn.euler, signature="euler_number(n)", doc="Euler number", category=K)
    api.function("catalan", cn.catalan, signature="catalan(n)", doc="Catalan number", category=K,
                 example="catalan(5)")
    api.function("bell", cn.bell, signature="bell(n)", doc="Bell number", category=K)
    api.function("stirling", cn.stirling, signature="stirling(n, k)", doc="Stirling number of the second kind",
                 category=K)
    api.function("partitions", cn.partition, signature="partitions(n)", doc="number of integer partitions",
                 category=K)
    api.function("factorial2", sp.factorial2, signature="factorial2(n)", doc="double factorial n!!", category=K)
    api.function("rising", sp.rf, signature="rising(x, n)", doc="rising factorial", category=K)
    api.function("falling", sp.ff, signature="falling(x, n)", doc="falling factorial", category=K)
    api.function("multinomial", lambda *ks: sp.multinomial_coefficients if False else
                 sp.factorial(sum(int(k) for k in ks)) / sp.Mul(*[sp.factorial(int(k)) for k in ks]),
                 signature="multinomial(k1, k2, ...)", doc="multinomial coefficient", category=K)
