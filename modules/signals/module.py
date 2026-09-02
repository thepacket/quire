"""Signals: sampling, spectra, windows, convolution, digital filters, Fourier coefficients."""
import numpy as np
import sympy as sp
from scipy import signal as sig

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "signals"
DESCRIPTION = "Sampling, FFT spectra, windows, convolution, IIR/FIR filters, Fourier coefficients."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _floats(xs, what="data"):
    if isinstance(xs, sp.MatrixBase):
        xs = list(xs)
    if not isinstance(xs, (list, tuple)):
        raise EvalError(f"{what} must be a list of numbers.")
    try:
        return np.array([float(U.strip_units(sp.sympify(v))[0]) for v in xs], dtype=float)
    except (TypeError, ValueError):
        raise EvalError(f"{what} must contain numbers only.") from None


def _list(a):
    return [sp.Float(float(v)) for v in a]


def sample_signal(expr, t, fs, duration):
    """Samples of an expression in t at rate fs for the given duration (plain numbers, seconds and Hz)."""
    if not isinstance(t, sp.Symbol):
        raise EvalError("Give the time variable, e.g. sample_signal(sin(2 pi 50 t), t, 1000, 0.1).")
    fs_v, T = U.si_value(fs, 1, "fs"), U.si_value(duration, 1, "duration")
    n = int(round(fs_v * T))
    ts = np.arange(n) / fs_v
    f = sp.lambdify(t, U.strip_units(sp.sympify(expr))[0], modules="numpy")
    with np.errstate(all="ignore"):
        ys = np.asarray(f(ts), dtype=float)
    if ys.shape != ts.shape:
        ys = np.full(ts.shape, float(ys))
    _note(f"{n} samples at {fs_v:g} Hz")
    return _list(ys)


def spectrum(samples, fs):
    """[frequencies, amplitudes]: single-sided amplitude spectrum of real samples (plot as scatter)."""
    x = _floats(samples)
    fs_v = U.si_value(fs, 1, "fs")
    n = x.size
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / fs_v)
    amp = np.abs(X) * 2 / n
    amp[0] /= 2
    if n % 2 == 0:
        amp[-1] /= 2
    _note(f"FFT of {n} samples, resolution {fs_v / n:g} Hz, Nyquist {fs_v / 2:g} Hz")
    return [_list(freqs), _list(amp)]


def power_spectrum_db(samples, fs):
    x = _floats(samples)
    fs_v = U.si_value(fs, 1, "fs")
    f, p = sig.periodogram(x, fs_v)
    return [_list(f), _list(10 * np.log10(np.maximum(p, 1e-300)))]


def hann(n):
    return _list(np.hanning(int(n)))


def hamming(n):
    return _list(np.hamming(int(n)))


def apply_window(samples, window):
    x, w = _floats(samples), _floats(window, "window")
    if x.size != w.size:
        raise EvalError("window and samples must have the same length.")
    return _list(x * w)


def convolve(a, b):
    return _list(np.convolve(_floats(a), _floats(b)))


def moving_average(samples, n):
    x = _floats(samples)
    n = int(n)
    return _list(np.convolve(x, np.ones(n) / n, mode="valid"))


def rms(samples):
    x = _floats(samples)
    return sp.Float(float(np.sqrt(np.mean(x ** 2))))


def snr_db(signal_samples, noise_samples):
    s, n = _floats(signal_samples), _floats(noise_samples, "noise")
    return sp.Float(10 * np.log10(np.mean(s ** 2) / np.mean(n ** 2)))


def nyquist_rate(f_max):
    return 2 * sp.sympify(f_max)


def aliased_frequency(f, fs):
    """Frequency observed after sampling f at fs."""
    f, fs = sp.sympify(f), sp.sympify(fs)
    if f.is_number and fs.is_number:
        fv, fsv = float(U.strip_units(f)[0]), float(U.strip_units(fs)[0])
        a = abs(fv - fsv * round(fv / fsv))
        unit = U.split_units(f)[1]
        return sp.Float(a) * unit
    return sp.Abs(f - fs * sp.floor(f / fs + sp.Rational(1, 2)))


def butter(order, fc, fs, highpass=False):
    """Butterworth IIR filter coefficients [b, a] (scipy)."""
    fc_v, fs_v = U.si_value(fc, 1, "fc"), U.si_value(fs, 1, "fs")
    b, a = sig.butter(int(order), fc_v / (fs_v / 2), btype="high" if highpass else "low")
    _note(f"Butterworth order {int(order)}, cutoff {fc_v:g} Hz at {fs_v:g} Hz sampling; use filter_apply(b, a, samples)")
    return [_list(b), _list(a)]


def fir_lowpass(taps, fc, fs):
    fc_v, fs_v = U.si_value(fc, 1, "fc"), U.si_value(fs, 1, "fs")
    return _list(sig.firwin(int(taps), fc_v / (fs_v / 2)))


def filter_apply(b, a, samples):
    y = sig.lfilter(_floats(b, "b"), _floats(a, "a"), _floats(samples))
    return _list(y)


def freq_response(b, a, fs, n=256):
    """[frequencies, gain in dB] of a digital filter (plot as scatter or line)."""
    fs_v = U.si_value(fs, 1, "fs")
    w, h = sig.freqz(_floats(b, "b"), _floats(a, "a"), worN=int(n))
    return [_list(w * fs_v / (2 * np.pi)), _list(20 * np.log10(np.maximum(np.abs(h), 1e-12)))]


def z_transfer(b, a, z):
    """H(z) from coefficient lists (b0 + b1 z^-1 + ...)/(a0 + a1 z^-1 + ...)."""
    num = sum(sp.sympify(c) * z ** (-k) for k, c in enumerate(b))
    den = sum(sp.sympify(c) * z ** (-k) for k, c in enumerate(a))
    return sp.simplify(num / den)


def impulse_response_dt(b, a, n=20):
    x = np.zeros(int(n))
    x[0] = 1
    return _list(sig.lfilter(_floats(b, "b"), _floats(a, "a"), x))


def fourier_coefficients(f, t, T, n=5):
    """[a0, [a1..an], [b1..bn]] of a T-periodic function on [-T/2, T/2], computed symbolically."""
    if not isinstance(t, sp.Symbol):
        raise EvalError("Give the time variable, e.g. fourier_coefficients(t^2, t, 2 pi, 3).")
    T = sp.sympify(T)
    w = 2 * sp.pi / T
    a0 = sp.simplify(2 / T * sp.integrate(f, (t, -T / 2, T / 2)))
    ak = [sp.simplify(2 / T * sp.integrate(f * sp.cos(k * w * t), (t, -T / 2, T / 2))) for k in range(1, int(n) + 1)]
    bk = [sp.simplify(2 / T * sp.integrate(f * sp.sin(k * w * t), (t, -T / 2, T / 2))) for k in range(1, int(n) + 1)]
    _note("f(t) = a0/2 + sum a_k cos(k w t) + b_k sin(k w t), w = 2 pi / T")
    return [a0, ak, bk]


def fourier_partial_sum(f, t, T, n=5):
    a0, ak, bk = fourier_coefficients(f, t, T, n)
    w = 2 * sp.pi / sp.sympify(T)
    return a0 / 2 + sum(ak[k - 1] * sp.cos(k * w * t) + bk[k - 1] * sp.sin(k * w * t) for k in range(1, int(n) + 1))


def register(api):
    S = "Signals: sampling & spectra"
    api.function("sample_signal", sample_signal, signature="sample_signal(expr, t, fs, duration)", doc="samples of a signal",
                 category=S, example="sample_signal(sin(2 pi 50 t) + 0.5 sin(2 pi 120 t), t, 1000, 0.5)")
    api.function("spectrum", spectrum, signature="spectrum(samples, fs)", doc="[f, amplitude] single-sided FFT; plot as scatter",
                 category=S, example="spectrum(x, 1000)")
    api.function("power_spectrum_db", power_spectrum_db, signature="power_spectrum_db(samples, fs)", doc="[f, dB] periodogram", category=S)
    api.function("hann", hann, signature="hann(n)", doc="Hann window", category=S)
    api.function("hamming", hamming, signature="hamming(n)", doc="Hamming window", category=S)
    api.function("apply_window", apply_window, signature="apply_window(samples, window)", doc="multiply by a window", category=S)
    api.function("nyquist_rate", nyquist_rate, signature="nyquist_rate(f_max)", doc="2 f_max", category=S)
    api.function("aliased_frequency", aliased_frequency, signature="aliased_frequency(f, fs)", doc="apparent frequency after sampling",
                 category=S, example="aliased_frequency(900 Hz, 1000 Hz)")
    P = "Signals: processing"
    api.function("convolve", convolve, signature="convolve(a, b)", doc="linear convolution", category=P)
    api.function("moving_average", moving_average, signature="moving_average(samples, n)", doc="n-point moving average", category=P)
    api.function("rms", rms, signature="rms(samples)", doc="root mean square", category=P)
    api.function("snr_db", snr_db, signature="snr_db(signal, noise)", doc="signal-to-noise ratio in dB", category=P)
    api.function("butter", butter, signature="butter(order, fc, fs, highpass)", doc="Butterworth IIR [b, a]", category=P,
                 example="butter(4, 100, 1000)")
    api.function("fir_lowpass", fir_lowpass, signature="fir_lowpass(taps, fc, fs)", doc="windowed-sinc FIR taps", category=P)
    api.function("filter_apply", filter_apply, signature="filter_apply(b, a, samples)", doc="run a digital filter", category=P)
    api.function("freq_response", freq_response, signature="freq_response(b, a, fs)", doc="[f, dB] of a digital filter", category=P)
    api.function("z_transfer", z_transfer, signature="z_transfer(b, a, z)", doc="H(z) from coefficients", category=P)
    api.function("impulse_response_dt", impulse_response_dt, signature="impulse_response_dt(b, a, n)", doc="first n samples of h[n]", category=P)
    F = "Signals: Fourier series"
    api.function("fourier_coefficients", fourier_coefficients, signature="fourier_coefficients(f, t, T, n)",
                 doc="[a0, [a_k], [b_k]] computed symbolically", category=F, example="fourier_coefficients(t, t, 2 pi, 3)")
    api.function("fourier_partial_sum", fourier_partial_sum, signature="fourier_partial_sum(f, t, T, n)",
                 doc="the n-term Fourier approximation as an expression", category=F, example="fourier_partial_sum(t^2, t, 2 pi, 4)")
