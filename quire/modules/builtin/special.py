"""Special functions."""
import sympy as sp


def register(api):
    S = "Special functions"
    api.function("gamma", sp.gamma, signature="gamma(x)", doc="gamma function", category=S, example="gamma(1/2)")
    api.function("lgamma", sp.loggamma, signature="lgamma(x)", doc="log of the gamma function", category=S)
    api.function("digamma", sp.digamma, signature="digamma(x)", doc="digamma ψ(x)", category=S)
    api.function("beta", sp.beta, signature="beta(a, b)", doc="beta function", category=S)
    api.function("erf", sp.erf, signature="erf(x)", doc="error function", category=S)
    api.function("erfc", sp.erfc, signature="erfc(x)", doc="complementary error function", category=S)
    api.function("erfinv", sp.erfinv, signature="erfinv(x)", doc="inverse error function", category=S)
    api.function("zeta", sp.zeta, signature="zeta(s)", doc="Riemann zeta", category=S, example="zeta(2)")
    api.function("polylog", sp.polylog, signature="polylog(s, z)", doc="polylogarithm", category=S)
    api.function("lambertw", sp.LambertW, signature="lambertw(x)", doc="Lambert W function", category=S)
    api.function("besselj", sp.besselj, signature="besselj(n, x)", doc="Bessel J", category=S)
    api.function("bessely", sp.bessely, signature="bessely(n, x)", doc="Bessel Y", category=S)
    api.function("besseli", sp.besseli, signature="besseli(n, x)", doc="modified Bessel I", category=S)
    api.function("besselk", sp.besselk, signature="besselk(n, x)", doc="modified Bessel K", category=S)
    api.function("airyai", sp.airyai, signature="airyai(x)", doc="Airy Ai", category=S)
    api.function("airybi", sp.airybi, signature="airybi(x)", doc="Airy Bi", category=S)
    api.function("Ei", sp.Ei, signature="Ei(x)", doc="exponential integral", category=S)
    api.function("Si", sp.Si, signature="Si(x)", doc="sine integral", category=S)
    api.function("Ci", sp.Ci, signature="Ci(x)", doc="cosine integral", category=S)
    api.function("li", sp.li, signature="li(x)", doc="logarithmic integral", category=S)
    api.function("elliptic_k", sp.elliptic_k, signature="elliptic_k(m)", doc="complete elliptic integral K",
                 category=S)
    api.function("elliptic_e", sp.elliptic_e, signature="elliptic_e(m)", doc="complete elliptic integral E",
                 category=S)
    api.function("heaviside", lambda x: sp.Heaviside(x), signature="heaviside(x)", doc="unit step", category=S)
    api.function("dirac", sp.DiracDelta, signature="dirac(x)", doc="Dirac delta", category=S)
    api.function("kronecker", sp.KroneckerDelta, signature="kronecker(i, j)", doc="Kronecker delta", category=S)
    O = "Orthogonal polynomials"
    api.function("legendre", sp.legendre, signature="legendre(n, x)", doc="Legendre polynomial", category=O,
                 example="legendre(2, x)")
    api.function("hermite", sp.hermite, signature="hermite(n, x)", doc="Hermite polynomial", category=O)
    api.function("chebyshevt", sp.chebyshevt, signature="chebyshevt(n, x)", doc="Chebyshev T", category=O)
    api.function("chebyshevu", sp.chebyshevu, signature="chebyshevu(n, x)", doc="Chebyshev U", category=O)
    api.function("laguerre", sp.laguerre, signature="laguerre(n, x)", doc="Laguerre polynomial", category=O)
    api.function("jacobi", sp.jacobi, signature="jacobi(n, a, b, x)", doc="Jacobi polynomial", category=O)
