"""Synthetic NEOWISE-like W1 light curves, following Paz (2024), Section 2.3,
Algorithms 2-6.

Classes (Section 3.1): 0 = null, 1 = transient, 2 = pulsator, 3 = transit.

Where the paper leaves a detail open, the choice made here is marked
"ASSUMPTION" so it can be revisited.
"""
from __future__ import annotations

import numpy as np

CLASSES = ["null", "transient", "pulsator", "transit"]
TOTAL_DAYS = 4000.0          # ~10.5 yr NEOWISE window (Section 2.2)
ZERO_POINT_JY = 309.54       # Algorithm 2 (Eq. 1 prints 309.25)


# --------------------------------------------------------------- helpers ---
def base_flux(rng: np.random.Generator) -> float:
    """Algorithm 2 BASEFLUX. Magnitude in [6, 16], weighted toward dimmer
    sources (footnote 8). ASSUMPTION: weighting m = 16 - 10*u^2."""
    m = 16.0 - 10.0 * rng.random() ** 2
    return ZERO_POINT_JY * 10 ** (-m / 2.5)


def uncertainty(rng: np.random.Generator) -> float:
    """Algorithm 2 UNCERTAINTY: sigma in [1e-4, 0.1]. ASSUMPTION: sampled
    log-uniformly so all orders of magnitude are represented."""
    return float(10 ** rng.uniform(-4, -1))


def time_sampling(rng: np.random.Generator, total_days: float = TOTAL_DAYS) -> np.ndarray:
    """Algorithm 2 TIMESAMPLING. Groups of 10-50 apparitions one orbit
    (0.115 d) apart, then a ~170 d gap until the scan returns.
    ASSUMPTION: the printed pseudocode has the two branches swapped (it would
    put a 170-day gap between almost every point); this follows the text."""
    per_view = int(rng.integers(10, 51))
    t = [0.0]
    while t[-1] < total_days:
        if len(t) % per_view == 0:
            t.append(t[-1] + 170.0)
        else:
            t.append(t[-1] + 0.115)
    return np.asarray(t[:-1])


# -------------------------------------------------------------- classes ---
def null_curve(t, w, s, rng):
    """Algorithm 3: constant source plus Gaussian noise."""
    return rng.normal(w, s, size=t.size)


def transient_curve(t, w, s, rng):
    """Algorithm 4. l(t) = 1/(tau s sqrt(2 pi)) exp(-(arsech(tau) - u)^2 / 2s^2).
    ASSUMPTION: tau = (t - t*)/d is the time since onset as a fraction of the
    event duration; with raw days the event would be invisibly small."""
    d = rng.uniform(600, 3000)
    t_star = rng.uniform(0, max(TOTAL_DAYS - d, 1.0))
    sp, u = rng.uniform(1, 2), rng.uniform(1, 3)
    y = np.full(t.size, w, dtype=float)
    on = (t > t_star) & (t < t_star + d)
    tau = np.clip((t[on] - t_star) / d, 1e-6, 1 - 1e-9)
    arsech = np.log((1 + np.sqrt(1 - tau ** 2)) / tau)
    lum = np.exp(-((arsech - u) ** 2) / (2 * sp ** 2)) / (tau * sp * np.sqrt(2 * np.pi))
    y[on] = w * lum + w
    y = rng.normal(y, s)
    keep = np.ones(t.size, bool)
    if rng.random() < 0.5:                     # visible only during the event
        keep = y > 1.25 * w
        if keep.sum() < 10:                    # ASSUMPTION: keep >= 10 points
            keep = np.ones(t.size, bool)
    return y, keep


def transit_curve(t, w, s, rng):
    """Algorithm 5: trapezoidal primary + shallower secondary eclipse.
    ASSUMPTIONS: trough (duty) length 2-15% of the period, ingress/egress 20%
    of the trough, secondary at phase 0.5; depth >= 3 S/N, capped at 0.6."""
    period = rng.uniform(0.1, 25)
    min_depth = min(3 * s / w, 0.55)
    depth = rng.uniform(min_depth, 0.6)
    sec = rng.uniform(0.2, 0.9)
    duty = rng.uniform(0.02, 0.15)
    ramp = 0.2 * duty
    phase = (t / period + rng.random()) % 1.0

    def trapezoid(p, centre):
        dist = np.abs(((p - centre + 0.5) % 1.0) - 0.5)
        half = duty / 2
        shape = np.clip((half - dist) / ramp, 0, 1)
        return shape

    dip = depth * trapezoid(phase, 0.0) + depth * sec * trapezoid(phase, 0.5)
    return rng.normal(w * (1 - dip), s)


def pulsator_curve(t, w, s, rng):
    """Algorithm 6: random Dirac impulses on a 101-point period grid,
    convolved with a Gaussian envelope, repeated over the window.
    ASSUMPTIONS: peak signs random (Figure 7a shows both); envelope width
    a in [20, 400]; amplitude is a fractional change of the base flux."""
    period = rng.uniform(0.1, 500)
    grid = np.zeros(101)
    n_peaks = int(rng.integers(1, 6))
    lo = min(3 * s / w, 0.7)
    heights = rng.uniform(lo, 0.75, n_peaks) * rng.choice([-1, 1], n_peaks)
    centres = rng.integers(0, 100, n_peaks)
    grid[centres] = heights
    a = rng.uniform(20, 400)
    x = np.arange(-50, 51)
    env = np.exp(-x ** 2 / a) / np.sqrt(a * np.pi)
    smooth = np.convolve(grid, env, mode="same")
    smooth *= np.max(np.abs(heights)) / max(np.max(np.abs(smooth)), 1e-12)
    pos = ((t / period + rng.random()) % 1.0) * 100
    return rng.normal(w * (1 + np.interp(pos, np.arange(101), smooth)), s)


# ----------------------------------------------------------------- main ---
def make_curve(label: int, rng: np.random.Generator):
    """Return (time_days, flux_jy, reported_err_jy) for one synthetic source."""
    t = time_sampling(rng)
    # The paper requires variability of at least 3 S/N (Algorithms 5-6).
    # ASSUMPTION: enforce 3*sigma/flux <= 0.5 for EVERY class (redrawing flux
    # and sigma together), so variable classes are detectable and all classes
    # share the same noise distribution (no shortcut via noise level).
    while True:
        w, s = base_flux(rng), uncertainty(rng)
        if 3 * s / w <= 0.5:
            break
    keep = np.ones(t.size, bool)
    if label == 0:
        y = null_curve(t, w, s, rng)
    elif label == 1:
        y, keep = transient_curve(t, w, s, rng)
    elif label == 2:
        y = pulsator_curve(t, w, s, rng)
    elif label == 3:
        y = transit_curve(t, w, s, rng)
    else:
        raise ValueError(label)
    # Section 2.3.1: reported uncertainty = sigma times a factor in [0.4, 0.6]
    err = np.full(t.size, s * rng.uniform(0.4, 0.6))
    return t[keep], y[keep], err[keep]
