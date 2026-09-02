"""Plane geometry on sympy.geometry: points, lines, circles, polygons, triangles, transformations.

Shapes are drawn with the "shapes" plot kind; equation_of gives the implicit equation
for the implicit plot kind. Angles are degrees.
"""
import sympy as sp
from sympy import geometry as g
from sympy.physics import units as u

from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "geometry"
DESCRIPTION = "Points, lines, circles, polygons, triangles: intersections, areas, centres, transformations."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _pt(P):
    if isinstance(P, g.Point):
        return P
    if isinstance(P, (list, tuple, sp.MatrixBase)) and len(P) == 2:
        return g.Point(*P)
    raise EvalError("Expected a point: point(x, y) or [x, y].")


def _deg(rad):
    x = sp.sympify(rad) * 180 / sp.pi
    if x.is_number and not x.free_symbols and not x.is_Rational:
        v = sp.N(x, 15)
        if abs(v - round(v)) < 1e-12:
            return sp.Integer(round(v)) * u.degree
        return sp.Float(v, 10) * u.degree
    return sp.simplify(x) * u.degree


def _rad(a):
    a = sp.sympify(a)
    from quire.engine import units as U
    return U.strip_angles(a) if U.has_units(a) else a * sp.pi / 180


def point(x, y):
    return g.Point(x, y)


def line(P, Q):
    return g.Line(_pt(P), _pt(Q))


def segment(P, Q):
    return g.Segment(_pt(P), _pt(Q))


def ray(P, Q):
    return g.Ray(_pt(P), _pt(Q))


def line_slope(P, slope):
    return g.Line(_pt(P), slope=sp.sympify(slope))


def circle(center, r):
    return g.Circle(_pt(center), sp.sympify(r))


def ellipse(center, a, b):
    return g.Ellipse(_pt(center), sp.sympify(a), sp.sympify(b))


def polygon(*pts):
    pts = pts[0] if len(pts) == 1 and isinstance(pts[0], (list, tuple)) and pts[0] and isinstance(pts[0][0], (list, tuple, g.Point)) else pts
    return g.Polygon(*[_pt(p) for p in pts])


def triangle(P, Q, R):
    return g.Triangle(_pt(P), _pt(Q), _pt(R))


def regular_polygon(center, r, n):
    return g.RegularPolygon(_pt(center), sp.sympify(r), int(n))


def intersect(a, b):
    res = g.intersection(a, b)
    return list(res)


def distance(a, b):
    return sp.simplify(a.distance(b) if hasattr(a, "distance") else _pt(a).distance(b))


def area(shape):
    return sp.simplify(sp.Abs(shape.area))


def perimeter(shape):
    if hasattr(shape, "perimeter"):
        return sp.simplify(shape.perimeter)
    if hasattr(shape, "circumference"):
        return sp.simplify(shape.circumference)
    raise EvalError("This shape has no perimeter.")


def centroid(shape):
    return shape.centroid if hasattr(shape, "centroid") else shape.center


def midpoint_of(P, Q):
    return _pt(P).midpoint(_pt(Q))


def angle_between(l1, l2):
    """Angle between two lines in degrees."""
    return _deg(l1.angle_between(l2))


def is_parallel(l1, l2):
    return l1.is_parallel(l2)


def is_perpendicular(l1, l2):
    return l1.is_perpendicular(l2)


def perpendicular_line(l, P):
    return l.perpendicular_line(_pt(P))


def parallel_line(l, P):
    return l.parallel_line(_pt(P))


def tangent_lines(shape, P):
    return list(shape.tangent_lines(_pt(P)))


def reflect(shape, l):
    return shape.reflect(l)


def rotate(shape, angle, center=None):
    return shape.rotate(_rad(angle), _pt(center) if center is not None else None)


def translate(shape, dx, dy):
    return shape.translate(sp.sympify(dx), sp.sympify(dy))


def scale_shape(shape, kx, ky=None, center=None):
    ky = kx if ky is None else ky
    return shape.scale(sp.sympify(kx), sp.sympify(ky), _pt(center) if center is not None else None)


def circumcircle(tri):
    return tri.circumcircle


def incircle(tri):
    return tri.incircle


def orthocenter(tri):
    return tri.orthocenter


def encloses(shape, P):
    return shape.encloses_point(_pt(P))


def convex_hull(*pts):
    pts = pts[0] if len(pts) == 1 and isinstance(pts[0], (list, tuple)) else pts
    return g.convex_hull(*[_pt(p) for p in pts])


def equation_of(shape, x, y):
    """Implicit equation F(x, y) == 0 of a line, circle or ellipse (for the implicit plot kind)."""
    eq = shape.equation(x, y)
    return sp.Eq(eq, 0)


def triangle_from_sides(a, b, c):
    """[A, B, C in degrees, area] of a triangle with sides a, b, c (law of cosines, Heron)."""
    a, b, c = (sp.sympify(v) for v in (a, b, c))
    A = sp.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    B = sp.acos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c))
    C = sp.pi - A - B
    s = (a + b + c) / 2
    ar = sp.sqrt(s * (s - a) * (s - b) * (s - c))
    _note("law of cosines for the angles, Heron's formula for the area")
    return [_deg(A), _deg(B), _deg(C), sp.simplify(ar)]


def law_of_cosines(a, b, C):
    """Third side from two sides and the included angle."""
    return sp.simplify(sp.sqrt(a ** 2 + b ** 2 - 2 * a * b * sp.cos(_rad(C))))


def law_of_sines(a, A, B):
    """Side opposite angle B, given side a opposite angle A."""
    return sp.simplify(a * sp.sin(_rad(B)) / sp.sin(_rad(A)))


def register(api):
    C = "Geometry: construct"
    api.function("point", point, signature="point(x, y)", doc="a point", category=C, example="point(1, 2)")
    api.function("line", line, signature="line(P, Q)", doc="infinite line through two points", category=C, example="line(point(0, 0), point(1, 1))")
    api.function("segment", segment, signature="segment(P, Q)", doc="segment", category=C)
    api.function("ray", ray, signature="ray(P, Q)", doc="ray from P through Q", category=C)
    api.function("line_slope", line_slope, signature="line_slope(P, slope)", doc="line through P with a slope", category=C)
    api.function("circle", circle, signature="circle(center, r)", doc="circle", category=C, example="circle(point(0, 0), 2)")
    api.function("ellipse", ellipse, signature="ellipse(center, a, b)", doc="ellipse with semi-axes a, b", category=C)
    api.function("polygon", polygon, signature="polygon(P1, P2, ...)", doc="polygon from points", category=C)
    api.function("triangle", triangle, signature="triangle(P, Q, R)", doc="triangle", category=C, example="triangle(point(0, 0), point(4, 0), point(0, 3))")
    api.function("regular_polygon", regular_polygon, signature="regular_polygon(center, r, n)", doc="regular n-gon", category=C)
    M = "Geometry: measure & relate"
    api.function("intersect", intersect, signature="intersect(a, b)", doc="intersection points/segments", category=M,
                 example="intersect(circle(point(0, 0), 1), line(point(0, 0), point(1, 1)))")
    api.function("distance", distance, signature="distance(a, b)", doc="distance between objects", category=M)
    api.function("area", area, signature="area(shape)", doc="area", category=M)
    api.function("perimeter", perimeter, signature="perimeter(shape)", doc="perimeter or circumference", category=M)
    api.function("centroid", centroid, signature="centroid(shape)", doc="centroid or centre", category=M)
    api.function("midpoint_of", midpoint_of, signature="midpoint_of(P, Q)", doc="midpoint", category=M)
    api.function("angle_between", angle_between, signature="angle_between(l1, l2)", doc="angle between lines (degrees)", category=M)
    api.function("is_parallel", is_parallel, signature="is_parallel(l1, l2)", doc="parallel?", category=M)
    api.function("is_perpendicular", is_perpendicular, signature="is_perpendicular(l1, l2)", doc="perpendicular?", category=M)
    api.function("encloses", encloses, signature="encloses(shape, P)", doc="is P strictly inside?", category=M)
    api.function("equation_of", equation_of, signature="equation_of(shape, x, y)", doc="implicit equation for the implicit plot", category=M,
                 example="equation_of(circle(point(1, 0), 2), x, y)")
    api.function("triangle_from_sides", triangle_from_sides, signature="triangle_from_sides(a, b, c)", doc="[A, B, C, area]", category=M,
                 example="triangle_from_sides(3, 4, 5)")
    api.function("law_of_cosines", law_of_cosines, signature="law_of_cosines(a, b, C)", doc="third side", category=M)
    api.function("law_of_sines", law_of_sines, signature="law_of_sines(a, A, B)", doc="side opposite B", category=M)
    T = "Geometry: transform & derive"
    api.function("perpendicular_line", perpendicular_line, signature="perpendicular_line(l, P)", doc="perpendicular through P", category=T)
    api.function("parallel_line", parallel_line, signature="parallel_line(l, P)", doc="parallel through P", category=T)
    api.function("tangent_lines", tangent_lines, signature="tangent_lines(circle, P)", doc="tangents from P", category=T)
    api.function("reflect", reflect, signature="reflect(shape, line)", doc="mirror image", category=T)
    api.function("rotate", rotate, signature="rotate(shape, angle, center)", doc="rotate by degrees", category=T)
    api.function("translate", translate, signature="translate(shape, dx, dy)", doc="shift", category=T)
    api.function("scale_shape", scale_shape, signature="scale_shape(shape, kx, ky, center)", doc="scale", category=T)
    api.function("circumcircle", circumcircle, signature="circumcircle(triangle)", doc="circumscribed circle", category=T)
    api.function("incircle", incircle, signature="incircle(triangle)", doc="inscribed circle", category=T)
    api.function("orthocenter", orthocenter, signature="orthocenter(triangle)", doc="orthocentre", category=T)
    api.function("convex_hull", convex_hull, signature="convex_hull([P1, P2, ...])", doc="convex hull polygon", category=T)
