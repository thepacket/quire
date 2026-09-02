"""Geodesy: distances and bearings on the Earth, WGS84 conversions, UTM projection.

Latitudes and longitudes are degrees (north and east positive) unless given with angle units.
"""
import math

import sympy as sp
from sympy.physics import units as u

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "geodesy"
DESCRIPTION = "Great-circle distance and bearing, WGS84 geodetic/ECEF, UTM, map scale."

A_WGS = 6378137.0
F_WGS = 1 / 298.257223563
E2 = F_WGS * (2 - F_WGS)
R_MEAN = 6371008.8


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _rad(a):
    a = sp.sympify(a)
    return float(U.strip_angles(a)) if U.has_units(a) else float(a) * math.pi / 180


def _deg(x):
    return sp.Float(math.degrees(x), 10) * u.degree


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance on a sphere of mean radius 6371 km."""
    p1, l1, p2, l2 = _rad(lat1), _rad(lon1), _rad(lat2), _rad(lon2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2
    _note("spherical Earth, mean radius 6371.0088 km; accurate to about 0.5 %")
    return sp.Float(2 * R_MEAN * math.asin(math.sqrt(h)), 10) * u.meter


def vincenty(lat1, lon1, lat2, lon2):
    """Geodesic distance on the WGS84 ellipsoid (Vincenty's inverse formula)."""
    p1, l1, p2, l2 = _rad(lat1), _rad(lon1), _rad(lat2), _rad(lon2)
    a, f = A_WGS, F_WGS
    b = a * (1 - f)
    U1, U2 = math.atan((1 - f) * math.tan(p1)), math.atan((1 - f) * math.tan(p2))
    L = l2 - l1
    lam = L
    for _ in range(200):
        sinl, cosl = math.sin(lam), math.cos(lam)
        sin_sigma = math.hypot(math.cos(U2) * sinl, math.cos(U1) * math.sin(U2) - math.sin(U1) * math.cos(U2) * cosl)
        if sin_sigma == 0:
            return sp.Float(0) * u.meter
        cos_sigma = math.sin(U1) * math.sin(U2) + math.cos(U1) * math.cos(U2) * cosl
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = math.cos(U1) * math.cos(U2) * sinl / sin_sigma
        cos2_alpha = 1 - sin_alpha ** 2
        cos_2sm = cos_sigma - 2 * math.sin(U1) * math.sin(U2) / cos2_alpha if cos2_alpha else 0.0
        C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
        lam_prev = lam
        lam = L + (1 - C) * f * sin_alpha * (sigma + C * sin_sigma * (cos_2sm + C * cos_sigma * (-1 + 2 * cos_2sm ** 2)))
        if abs(lam - lam_prev) < 1e-12:
            break
    else:
        raise EvalError("Vincenty did not converge (nearly antipodal points); use haversine.")
    u2 = cos2_alpha * (a ** 2 - b ** 2) / b ** 2
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    d_sigma = B * sin_sigma * (cos_2sm + B / 4 * (cos_sigma * (-1 + 2 * cos_2sm ** 2) - B / 6 * cos_2sm * (-3 + 4 * sin_sigma ** 2) * (-3 + 4 * cos_2sm ** 2)))
    _note("WGS84 ellipsoid, Vincenty inverse; millimetre accuracy")
    return sp.Float(b * A * (sigma - d_sigma), 12) * u.meter


def bearing(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    p1, l1, p2, l2 = _rad(lat1), _rad(lon1), _rad(lat2), _rad(lon2)
    y = math.sin(l2 - l1) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(l2 - l1)
    return _deg(math.atan2(y, x) % (2 * math.pi))


def destination(lat, lon, bearing_deg, distance):
    """[lat, lon] reached by travelling the distance along the initial bearing (sphere)."""
    p1, l1, th = _rad(lat), _rad(lon), _rad(bearing_deg)
    d = U.si_value(distance, u.meter, "distance") / R_MEAN
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(th))
    l2 = l1 + math.atan2(math.sin(th) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return [_deg(p2), _deg((l2 + math.pi) % (2 * math.pi) - math.pi)]


def midpoint(lat1, lon1, lat2, lon2):
    p1, l1, p2, l2 = _rad(lat1), _rad(lon1), _rad(lat2), _rad(lon2)
    bx = math.cos(p2) * math.cos(l2 - l1)
    by = math.cos(p2) * math.sin(l2 - l1)
    pm = math.atan2(math.sin(p1) + math.sin(p2), math.sqrt((math.cos(p1) + bx) ** 2 + by ** 2))
    lm = l1 + math.atan2(by, math.cos(p1) + bx)
    return [_deg(pm), _deg(lm)]


def geodetic_to_ecef(lat, lon, height=0):
    """[x, y, z] WGS84 Earth-centred coordinates."""
    p, l = _rad(lat), _rad(lon)
    h = U.si_value(height, u.meter, "height") if sp.sympify(height) != 0 else 0.0
    N = A_WGS / math.sqrt(1 - E2 * math.sin(p) ** 2)
    x = (N + h) * math.cos(p) * math.cos(l)
    y = (N + h) * math.cos(p) * math.sin(l)
    z = (N * (1 - E2) + h) * math.sin(p)
    return [sp.Float(v, 12) * u.meter for v in (x, y, z)]


def ecef_to_geodetic(x, y, z):
    """[lat, lon, height] from WGS84 ECEF coordinates (iterative)."""
    xv, yv, zv = (U.si_value(v, u.meter, n) for v, n in ((x, "x"), (y, "y"), (z, "z")))
    lon = math.atan2(yv, xv)
    p = math.hypot(xv, yv)
    lat = math.atan2(zv, p * (1 - E2))
    for _ in range(10):
        N = A_WGS / math.sqrt(1 - E2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - N
        lat = math.atan2(zv, p * (1 - E2 * N / (N + h)))
    N = A_WGS / math.sqrt(1 - E2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - N
    return [_deg(lat), _deg(lon), sp.Float(h, 10) * u.meter]


def utm(lat, lon):
    """[zone, easting, northing] on the UTM projection (WGS84); southern hemisphere adds 10 000 km."""
    p, l = _rad(lat), _rad(lon)
    lon_deg = math.degrees(l)
    zone = int((lon_deg + 180) // 6) + 1
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    k0 = 0.9996
    e2 = E2
    ep2 = e2 / (1 - e2)
    N = A_WGS / math.sqrt(1 - e2 * math.sin(p) ** 2)
    T = math.tan(p) ** 2
    C = ep2 * math.cos(p) ** 2
    A = (l - lon0) * math.cos(p)
    M = A_WGS * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * p
                 - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * p)
                 + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * p)
                 - (35 * e2 ** 3 / 3072) * math.sin(6 * p))
    easting = k0 * N * (A + (1 - T + C) * A ** 3 / 6 + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000
    northing = k0 * (M + N * math.tan(p) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                                             + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    if p < 0:
        northing += 10000000
    _note(f"UTM zone {zone}{'N' if p >= 0 else 'S'}, WGS84")
    return [sp.Integer(zone), sp.Float(easting, 10) * u.meter, sp.Float(northing, 10) * u.meter]


def map_scale_distance(map_length, scale):
    """Ground distance from a map length and a scale denominator (1:scale)."""
    return sp.simplify(map_length * scale)


def register(api):
    Gd = "Geodesy"
    api.function("haversine", haversine, signature="haversine(lat1, lon1, lat2, lon2)", doc="great-circle distance (sphere)", category=Gd,
                 example="haversine(45.5, -73.6, 51.5, -0.1) -> km")
    api.function("vincenty", vincenty, signature="vincenty(lat1, lon1, lat2, lon2)", doc="geodesic distance (WGS84)", category=Gd)
    api.function("bearing", bearing, signature="bearing(lat1, lon1, lat2, lon2)", doc="initial bearing (degrees)", category=Gd)
    api.function("destination", destination, signature="destination(lat, lon, bearing, distance)", doc="[lat, lon] after travelling", category=Gd,
                 example="destination(45.5, -73.6, 90, 100 km)")
    api.function("midpoint", midpoint, signature="midpoint(lat1, lon1, lat2, lon2)", doc="great-circle midpoint", category=Gd)
    api.function("geodetic_to_ecef", geodetic_to_ecef, signature="geodetic_to_ecef(lat, lon, h)", doc="[x, y, z] WGS84", category=Gd)
    api.function("ecef_to_geodetic", ecef_to_geodetic, signature="ecef_to_geodetic(x, y, z)", doc="[lat, lon, h] WGS84", category=Gd)
    api.function("utm", utm, signature="utm(lat, lon)", doc="[zone, easting, northing]", category=Gd, example="utm(45.5, -73.6)")
    api.function("map_scale_distance", map_scale_distance, signature="map_scale_distance(length, scale)", doc="ground distance from a 1:scale map", category=Gd)
