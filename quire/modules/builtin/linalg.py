"""Matrices and linear algebra."""
import sympy as sp

from ._util import as_list, matrix


def _eigenvects(A):
    out = []
    for val, mult, vecs in matrix(A).eigenvects():
        out.append([val, mult, [sp.ImmutableMatrix(v) for v in vecs]])
    return out


def _diagonalize(A):
    P, D = matrix(A).diagonalize()
    return [sp.ImmutableMatrix(P), sp.ImmutableMatrix(D)]


def _lu(A):
    L, Uu, _ = sp.Matrix(A).LUdecomposition()
    return [sp.ImmutableMatrix(L), sp.ImmutableMatrix(Uu)]


def _qr(A):
    Q, R = sp.Matrix(A).QRdecomposition()
    return [sp.ImmutableMatrix(Q), sp.ImmutableMatrix(R)]


def _solve_linear(A, b):
    bm = sp.Matrix(as_list(b)) if not isinstance(b, sp.MatrixBase) else b
    return sp.ImmutableMatrix(sp.Matrix(A).LUsolve(bm))


def register(api):
    M = "Matrices"
    api.function("matrix", lambda rows: sp.ImmutableMatrix(rows), signature="matrix([[a, b], [c, d]])",
                 doc="build a matrix; a flat list gives a column vector", category=M,
                 example="matrix([[1, 2], [3, 4]])")
    api.function("identity", sp.eye, signature="identity(n)", doc="n×n identity matrix", category=M)
    api.function("zeros", sp.zeros, signature="zeros(n, m)", doc="n×m zero matrix", category=M)
    api.function("ones", sp.ones, signature="ones(n, m)", doc="n×m matrix of ones", category=M)
    api.function("diag", lambda *d: sp.diag(*(as_list(d[0]) if len(d) == 1 else d)), signature="diag(a, b, c)",
                 doc="diagonal matrix", category=M)
    api.function("det", lambda A: matrix(A).det(), signature="det(A)", doc="determinant", category=M)
    api.function("inv", lambda A: matrix(A).inv(), signature="inv(A)", doc="inverse", category=M)
    api.function("transpose", lambda A: matrix(A).T, signature="transpose(A)", doc="transpose", category=M)
    api.function("trace", lambda A: matrix(A).trace(), signature="trace(A)", doc="trace", category=M)
    api.function("rank", lambda A: matrix(A).rank(), signature="rank(A)", doc="rank", category=M)
    api.function("rref", lambda A: sp.ImmutableMatrix(matrix(A).rref()[0]), signature="rref(A)",
                 doc="reduced row echelon form", category=M)
    api.function("nullspace", lambda A: [sp.ImmutableMatrix(v) for v in matrix(A).nullspace()],
                 signature="nullspace(A)", doc="basis of the null space", category=M)
    api.function("columnspace", lambda A: [sp.ImmutableMatrix(v) for v in matrix(A).columnspace()],
                 signature="columnspace(A)", doc="basis of the column space", category=M)
    api.function("eigenvals", lambda A: [v for v, m in matrix(A).eigenvals().items() for _ in range(m)],
                 signature="eigenvals(A)", doc="eigenvalues (with multiplicity)", category=M)
    api.function("eigenvects", _eigenvects, signature="eigenvects(A)",
                 doc="[eigenvalue, multiplicity, [vectors]] for each eigenvalue", category=M)
    api.function("diagonalize", _diagonalize, signature="diagonalize(A)", doc="[P, D] with A = P D P⁻¹", category=M)
    api.function("charpoly", lambda A, lam: matrix(A).charpoly(lam).as_expr(), signature="charpoly(A, lambda)",
                 doc="characteristic polynomial", category=M, example="charpoly(matrix([[2, 1], [1, 2]]), lambda_)")
    api.function("lu", _lu, signature="lu(A)", doc="[L, U] decomposition", category=M)
    api.function("qr", _qr, signature="qr(A)", doc="[Q, R] decomposition", category=M)
    api.function("expm", lambda A: sp.ImmutableMatrix(matrix(A).exp()).applyfunc(
        lambda e: sp.simplify(e.rewrite(sp.cos))), signature="expm(A)",
                 doc="matrix exponential", category=M, example="expm(matrix([[0, 1], [-1, 0]]) t)")
    api.function("solve_linear", _solve_linear, signature="solve_linear(A, b)", doc="solve A x = b", category=M,
                 example="solve_linear(matrix([[2, 1], [1, 3]]), [3, 5])")
    api.function("dot", lambda a, b: sp.Matrix(as_list(a)).dot(sp.Matrix(as_list(b))), signature="dot(a, b)",
                 doc="dot product", category=M)
    api.function("cross", lambda a, b: sp.ImmutableMatrix(sp.Matrix(as_list(a)).cross(sp.Matrix(as_list(b)))),
                 signature="cross(a, b)", doc="cross product", category=M)
    api.function("norm", lambda a: sp.Matrix(as_list(a)).norm(), signature="norm(a)", doc="Euclidean length",
                 category=M)
