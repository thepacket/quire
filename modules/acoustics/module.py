"""Music and acoustics: tuning systems and notes, rooms and reverberation, sound levels, filters and spectra on audio data."""
from __future__ import annotations

import math
import re

import numpy as np
import sympy as sp
from sympy.physics import units as u

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "acoustics"
DESCRIPTION = "Tuning systems, notes and cents, room modes and reverberation, decibels, audio filters and spectra."

NOTE_INDEX = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
JUST_RATIOS = [sp.Rational(1), sp.Rational(16, 15), sp.Rational(9, 8), sp.Rational(6, 5), sp.Rational(5, 4), sp.Rational(4, 3),
               sp.Rational(45, 32), sp.Rational(3, 2), sp.Rational(8, 5), sp.Rational(5, 3), sp.Rational(9, 5), sp.Rational(15, 8)]


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _hz(v, what="frequency"):
    return U.si_value(v, u.hertz, what) if U.has_units(v) else float(v)


def _floats(xs, what="samples"):
    if isinstance(xs, sp.MatrixBase):
        xs = list(xs)
    if not isinstance(xs, (list, tuple)):
        raise EvalError(f"{what} must be a list of numbers.")
    try:
        return np.array([float(U.strip_units(sp.sympify(x))[0]) for x in xs])
    except (TypeError, ValueError):
        raise EvalError(f"{what} must contain numbers only.") from None


# ---- tuning and notes
def midi_number(note):
    """MIDI number of a note name such as A4, Cs4 (C sharp), Bb3 (B flat)."""
    m = re.fullmatch(r"([A-G])(s|#|b|f)?(-?\d)", str(note))
    if not m:
        raise EvalError("Write a note as letter, optional s (sharp) or b (flat), octave: A4, Cs4, Bb3.")
    semis = NOTE_INDEX[m.group(1)] + {None: 0, "s": 1, "#": 1, "b": -1, "f": -1}[m.group(2)]
    return sp.Integer(12 * (int(m.group(3)) + 1) + semis)


def note_frequency(note, a4=sp.Integer(440) * u.hertz):
    """Equal-temperament frequency of a note name (A4 = 440 Hz unless told otherwise)."""
    n = midi_number(note)
    return sp.Float(_hz(a4) * 2 ** ((int(n) - 69) / 12), 6) * u.hertz


def midi_frequency(n, a4=sp.Integer(440) * u.hertz):
    return sp.Float(_hz(a4) * 2 ** ((float(n) - 69) / 12), 6) * u.hertz


def note_name(f):
    """Nearest equal-temperament note of a frequency, with the offset in cents."""
    fv = _hz(f)
    if fv <= 0:
        raise EvalError("Frequency must be positive.")
    n = 69 + 12 * math.log2(fv / 440)
    k = round(n)
    name = f"{NOTE_NAMES[k % 12]}{k // 12 - 1}"
    _note(f"{name} {100 * (n - k):+.1f} cents")
    return sp.Symbol(name.replace("#", "s"))


def cents(f1, f2):
    """Interval between two frequencies (or a ratio) in cents."""
    r = sp.sympify(f2) / sp.sympify(f1)
    return sp.Float(1200 * math.log2(float(U.strip_units(r)[0])), 6)


def equal_temperament(n, base=sp.Integer(1)):
    """Frequency ratio of n semitones (times a base frequency)."""
    return sp.simplify(sp.sympify(base) * sp.Integer(2) ** sp.Rational(int(n), 12))


def just_intonation(degree, base=sp.Integer(1)):
    """Five-limit just ratio of scale degree 0..12 (chromatic) times a base frequency."""
    d = int(degree)
    return sp.sympify(base) * JUST_RATIOS[d % 12] * sp.Integer(2) ** (d // 12)


def pythagorean(n, base=sp.Integer(1)):
    """Pythagorean ratio n fifths up (reduced to one octave) times a base frequency."""
    r = sp.Rational(3, 2) ** int(n)
    while r >= 2:
        r /= 2
    while r < 1:
        r *= 2
    return sp.sympify(base) * r


def scale(root, mode=sp.Symbol("major")):
    """Frequencies of a diatonic scale from a root note (major, minor, dorian, mixolydian, pentatonic, blues)."""
    patterns = {"major": [0, 2, 4, 5, 7, 9, 11, 12], "minor": [0, 2, 3, 5, 7, 8, 10, 12], "dorian": [0, 2, 3, 5, 7, 9, 10, 12],
                "mixolydian": [0, 2, 4, 5, 7, 9, 10, 12], "pentatonic": [0, 2, 4, 7, 9, 12], "blues": [0, 3, 5, 6, 7, 10, 12]}
    p = patterns.get(str(mode))
    if p is None:
        raise EvalError(f"Unknown mode '{mode}'. Use " + ", ".join(patterns) + ".")
    n = int(midi_number(root))
    return [midi_frequency(n + k) for k in p]


# ---- rooms and levels
def speed_of_sound(T=sp.Integer(20) * u.deg_C if hasattr(u, "deg_C") else sp.Integer(293) * u.kelvin):
    """Speed of sound in air, 331.3 sqrt(1 + T/273.15) m/s (T in °C or K)."""
    t = sp.sympify(T)
    if U.has_units(t):
        tk = U.si_value(t, u.kelvin, "temperature")
        tc = tk - 273.15
    else:
        tc = float(t)
    return sp.Float(331.3 * math.sqrt(1 + tc / 273.15), 6) * u.meter / u.second


def sound_wavelength(f, c=sp.Float(343) * u.meter / u.second):
    return sp.simplify(c / f)


def room_modes(L, W, H, count=10, c=sp.Float(343) * u.meter / u.second):
    """Lowest standing-wave frequencies of a rectangular room: [f, nx, ny, nz] rows (Rayleigh)."""
    dims = [U.si_value(v, u.meter, "room dimension") if U.has_units(v) else float(v) for v in (L, W, H)]
    cv = U.si_value(c, u.meter / u.second, "speed of sound") if U.has_units(c) else float(c)
    modes = []
    for nx in range(0, 6):
        for ny in range(0, 6):
            for nz in range(0, 6):
                if nx == ny == nz == 0:
                    continue
                f = cv / 2 * math.sqrt((nx / dims[0]) ** 2 + (ny / dims[1]) ** 2 + (nz / dims[2]) ** 2)
                modes.append((f, nx, ny, nz))
    modes.sort()
    n = max(1, int(count))
    kinds = {1: "axial", 2: "tangential", 3: "oblique"}
    _note("mode types: " + ", ".join(f"{f:.1f} Hz {kinds[sum(1 for k in (nx, ny, nz) if k)]}" for f, nx, ny, nz in modes[:min(n, 6)]))
    return sp.ImmutableMatrix([[sp.Float(f, 5), nx, ny, nz] for f, nx, ny, nz in modes[:n]])


def schroeder_frequency(rt60, volume):
    """Above about 2000 sqrt(T60 / V) Hz the room's modes overlap."""
    t = U.si_value(rt60, u.second, "RT60") if U.has_units(rt60) else float(rt60)
    v = U.si_value(volume, u.meter ** 3, "volume") if U.has_units(volume) else float(volume)
    return sp.Float(2000 * math.sqrt(t / v), 5) * u.hertz


def sabine_rt60(volume, absorption):
    """Reverberation time 0.161 V / A (A = total absorption in m^2 sabins)."""
    v = U.si_value(volume, u.meter ** 3, "volume") if U.has_units(volume) else float(volume)
    a = U.si_value(absorption, u.meter ** 2, "absorption") if U.has_units(absorption) else float(absorption)
    return sp.Float(0.161 * v / a, 5) * u.second


def absorption_area(surfaces):
    """Sum of area x coefficient over [[area, alpha], ...] rows."""
    rows = surfaces.tolist() if isinstance(surfaces, sp.MatrixBase) else surfaces
    return sp.simplify(sum(sp.sympify(r[0]) * sp.sympify(r[1]) for r in rows))


def spl(pressure, reference=sp.Float(20e-6) * u.pascal):
    """Sound pressure level 20 log10(p / 20 µPa) dB."""
    r = float(U.strip_units(sp.sympify(pressure) / reference)[0]) if U.has_units(pressure) else float(pressure) / 20e-6
    return sp.Float(20 * math.log10(r), 6)


def spl_sum(levels):
    """Combined level of incoherent sources: 10 log10(sum 10^(L/10))."""
    ls = _floats(levels, "levels")
    return sp.Float(10 * math.log10(np.sum(10 ** (ls / 10))), 6)


def spl_at_distance(L1, r1, r2):
    """Level at r2 given L1 at r1 (inverse square): L1 - 20 log10(r2/r1)."""
    ratio = float(U.strip_units(sp.sympify(r2) / sp.sympify(r1))[0])
    return sp.Float(float(L1) - 20 * math.log10(ratio), 6)


def a_weighting(f):
    """A-weighting correction in dB at frequency f (IEC 61672)."""
    fv = _hz(f)
    f2 = fv ** 2
    ra = 12194 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * math.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2))
    return sp.Float(20 * math.log10(ra) + 2.0, 5)


# ---- filters and spectra on audio data
def _scipy_signal():
    from scipy import signal

    return signal


def biquad(kind, fc, fs, Q=sp.Float(0.7071)):
    """Digital biquad coefficients [b0, b1, b2, a0, a1, a2] (Audio EQ Cookbook): lowpass, highpass, bandpass, notch."""
    k = str(kind)
    w0 = 2 * math.pi * _hz(fc) / _hz(fs)
    q = float(Q)
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2 * q)
    if k == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
    elif k == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
    elif k == "bandpass":
        b = [alpha, 0, -alpha]
    elif k == "notch":
        b = [1, -2 * cw, 1]
    else:
        raise EvalError("kind is lowpass, highpass, bandpass or notch.")
    a = [1 + alpha, -2 * cw, 1 - alpha]
    return [sp.Float(v, 8) for v in b + a]


def apply_filter(coefficients, samples):
    """Filter a list of samples with biquad coefficients (or [b..., a...] of equal length)."""
    c = _floats(coefficients, "coefficients")
    if len(c) == 6:
        b, a = c[:3], c[3:]
    elif len(c) % 2 == 0:
        b, a = c[: len(c) // 2], c[len(c) // 2:]
    else:
        raise EvalError("Coefficients are [b0, b1, b2, a0, a1, a2].")
    y = _scipy_signal().lfilter(b, a, _floats(samples))
    return [sp.Float(v, 8) for v in y]


def filter_response(coefficients, fs, n=200):
    """Magnitude response of a biquad as [f, dB] rows, for plotting."""
    c = _floats(coefficients, "coefficients")
    w, h = _scipy_signal().freqz(c[:3], c[3:], worN=int(n))
    f = w / (2 * math.pi) * _hz(fs)
    return sp.ImmutableMatrix([[sp.Float(fv, 6), sp.Float(20 * math.log10(max(abs(hv), 1e-12)), 6)] for fv, hv in zip(f, h)])


def audio_spectrum(samples, fs):
    """Amplitude spectrum of a signal as [f, amplitude] rows (Hann window)."""
    x = _floats(samples)
    n = len(x)
    if n < 4:
        raise EvalError("Need at least 4 samples.")
    w = np.hanning(n)
    X = np.abs(np.fft.rfft(x * w)) * 2 / w.sum()
    f = np.fft.rfftfreq(n, 1 / _hz(fs))
    return sp.ImmutableMatrix([[sp.Float(fv, 6), sp.Float(av, 6)] for fv, av in zip(f, X)])


def dominant_frequencies(samples, fs, count=3):
    """The strongest peaks of the spectrum, [f, amplitude] rows."""
    M = audio_spectrum(samples, fs)
    f = np.array([float(M[i, 0]) for i in range(M.rows)])
    a = np.array([float(M[i, 1]) for i in range(M.rows)])
    peaks = [i for i in range(1, len(a) - 1) if a[i] > a[i - 1] and a[i] >= a[i + 1]]
    peaks.sort(key=lambda i: -a[i])
    return [[sp.Float(f[i], 6) * u.hertz, sp.Float(a[i], 6)] for i in peaks[: int(count)]]


def tone(f, fs, duration, amplitude=1):
    """Samples of a sine tone, for trying filters and spectra."""
    n = int(round(_hz(fs) * (U.si_value(duration, u.second, "duration") if U.has_units(duration) else float(duration))))
    t = np.arange(n) / _hz(fs)
    return [sp.Float(v, 8) for v in float(amplitude) * np.sin(2 * math.pi * _hz(f) * t)]


def register(api):
    T = "Music: tuning & notes"
    api.function("note_frequency", note_frequency, signature="note_frequency(A4)", doc="equal-temperament frequency of a note (Cs4 = C#4, Bb3 = B♭3)", category=T,
                 example="note_frequency(Cs4)")
    api.function("midi_number", midi_number, signature="midi_number(A4)", doc="MIDI note number", category=T)
    api.function("midi_frequency", midi_frequency, signature="midi_frequency(69)", doc="frequency of a MIDI number", category=T)
    api.function("note_name", note_name, signature="note_name(f)", doc="nearest note and the offset in cents", category=T, example="note_name(450 Hz)")
    api.function("cents", cents, signature="cents(f1, f2)", doc="interval in cents", category=T, example="cents(1, 3/2)")
    api.function("equal_temperament", equal_temperament, signature="equal_temperament(n, base)", doc="ratio of n semitones", category=T)
    api.function("just_intonation", just_intonation, signature="just_intonation(degree, base)", doc="five-limit just ratio of a chromatic degree", category=T)
    api.function("pythagorean", pythagorean, signature="pythagorean(n, base)", doc="ratio after n fifths, reduced to the octave", category=T)
    api.function("scale", scale, signature="scale(root, mode)", doc="frequencies of a scale (major, minor, dorian, mixolydian, pentatonic, blues)", category=T,
                 example="scale(C4, major)")
    for m in ("major", "minor", "dorian", "mixolydian", "pentatonic", "blues"):
        api.constant(m, sp.Symbol(m), doc="scale mode", category=T)
    R = "Acoustics: rooms & levels"
    api.function("speed_of_sound", speed_of_sound, signature="speed_of_sound(T)", doc="in air at temperature T", category=R, example="speed_of_sound(20)")
    api.function("sound_wavelength", sound_wavelength, signature="sound_wavelength(f, c)", doc="c / f", category=R)
    api.function("room_modes", room_modes, signature="room_modes(L, W, H, count)", doc="lowest standing waves of a rectangular room [f, nx, ny, nz]", category=R,
                 example="room_modes(5 m, 4 m, 2.7 m, 8)")
    api.function("schroeder_frequency", schroeder_frequency, signature="schroeder_frequency(RT60, V)", doc="where modes start to overlap", category=R)
    api.function("sabine_rt60", sabine_rt60, signature="sabine_rt60(V, A)", doc="reverberation time 0.161 V / A", category=R, example="sabine_rt60(54 m^3, 12 m^2)")
    api.function("absorption_area", absorption_area, signature="absorption_area([[area, alpha], ...])", doc="total absorption", category=R)
    api.function("spl", spl, signature="spl(p)", doc="sound pressure level re 20 µPa", category=R, example="spl(1 Pa)")
    api.function("spl_sum", spl_sum, signature="spl_sum([L1, L2, ...])", doc="combined level of independent sources", category=R, example="spl_sum([80, 80])")
    api.function("spl_at_distance", spl_at_distance, signature="spl_at_distance(L1, r1, r2)", doc="inverse-square level change", category=R)
    api.function("a_weighting", a_weighting, signature="a_weighting(f)", doc="A-weighting in dB", category=R, example="a_weighting(100 Hz)")
    F = "Audio: filters & spectra"
    api.function("biquad", biquad, signature="biquad(lowpass, fc, fs, Q)", doc="biquad coefficients [b0, b1, b2, a0, a1, a2]", category=F,
                 example="biquad(lowpass, 1 kHz, 44.1 kHz)")
    for k in ("lowpass", "highpass", "bandpass", "notch"):
        api.constant(k, sp.Symbol(k), doc="filter kind for biquad", category=F)
    api.function("apply_filter", apply_filter, signature="apply_filter(coefficients, samples)", doc="run samples through the filter", category=F)
    api.function("filter_response", filter_response, signature="filter_response(coefficients, fs)", doc="[f, dB] rows of the magnitude response", category=F)
    api.function("audio_spectrum", audio_spectrum, signature="audio_spectrum(samples, fs)", doc="[f, amplitude] rows (Hann window)", category=F)
    api.function("dominant_frequencies", dominant_frequencies, signature="dominant_frequencies(samples, fs, count)", doc="strongest spectral peaks", category=F,
                 example="dominant_frequencies(tone(440 Hz, 8 kHz, 0.5 s), 8 kHz)")
    api.function("tone", tone, signature="tone(f, fs, duration, amplitude)", doc="samples of a sine tone", category=F)
