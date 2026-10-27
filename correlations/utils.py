"""Numba kernels for the spectrum and the original quadrature reference.

The reference still performs adaptive angular quadrature over a trapezoidal
radial integral. These kernels preserve the previous Cython formulas and
summation order without requiring an extension build or fast-math arithmetic.
"""

from importlib.resources import files
from math import cos, exp, pi, sin, sqrt

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _power_spectrum(value, k, pk):
    if value > k[-1] or value < k[0]:
        return 0.0
    left, right = 0, len(k) - 1
    while left + 1 < right:
        middle = (left + right) // 2
        if value < k[middle]:
            right = middle
        else:
            left = middle
    return (value - k[left]) / (k[right] - k[left]) * (pk[right] - pk[left]) + pk[left]


@njit(cache=True, nogil=True)
def _sigma(order, radius, k, pk):
    integral = 0.0
    previous = 0.0
    for j in range(len(k)):
        value = k[j] ** (2 + 2 * order) * pk[j] * exp(-((k[j] * radius) ** 2))
        if j:
            integral += (previous + value) / 2 * (k[j] - k[j - 1])
        previous = value
    return sqrt(integral / (2 * pi**2))


@njit(cache=True, nogil=True, error_model="numpy", inline="always")
def _integrand_value(
    k,
    k2pk,
    sin_theta,
    cos_theta,
    cos_phi,
    sin_phi,
    ikx,
    iky,
    ikz,
    ikk,
    dx,
    dy,
    dz,
    R1,
    R2,
):
    kx = k * sin_theta * cos_phi
    ky = k * sin_theta * sin_phi
    kz = k * cos_theta
    phase = kx * dx + ky * dy + kz * dz - (ikx + iky + ikz - ikk) * pi / 2
    value = k2pk * sin_theta * cos(phase)
    if R1 > 0:
        value *= exp(-((k * R1) ** 2) / 2)
    if R2 > 0:
        value *= exp(-((k * R2) ** 2) / 2)
    if ikx:
        value *= kx**ikx
    if iky:
        value *= ky**iky
    if ikz:
        value *= kz**ikz
    if ikk:
        value /= k**ikk
    return value


@njit(cache=True, nogil=True, error_model="numpy")
def _integrand(k, phi, theta, ikx, iky, ikz, ikk, dx, dy, dz, R1, R2, grid, pk):
    value = _integrand_value(
        k,
        k**2 * _power_spectrum(k, grid, pk),
        sin(theta),
        cos(theta),
        cos(phi),
        sin(phi),
        ikx,
        iky,
        ikz,
        ikk,
        dx,
        dy,
        dz,
        R1,
        R2,
    )
    return value / (8 * pi**3)


@njit(cache=True, nogil=True, error_model="numpy")
def _integrand_radial(phi, theta, ikx, iky, ikz, ikk, dx, dy, dz, R1, R2, k, k2pk):
    sin_theta, cos_theta = sin(theta), cos(theta)
    cos_phi, sin_phi = cos(phi), sin(phi)
    integral, previous = 0.0, 0.0
    for j in range(len(k)):
        value = _integrand_value(
            k[j],
            k2pk[j],
            sin_theta,
            cos_theta,
            cos_phi,
            sin_phi,
            ikx,
            iky,
            ikz,
            ikk,
            dx,
            dy,
            dz,
            R1,
            R2,
        )
        if j:
            integral += (k[j] - k[j - 1]) * (previous + value) / 2
        previous = value
    return integral / (8 * pi**3)


class Utils:
    """Spectrum interpolation, spectral moments, and quadrature integrands."""

    def __init__(self, k=None, Pk=None):
        if k is None and Pk is None:
            with (
                files("correlations").joinpath("data/power.dat").open("rb") as spectrum
            ):
                k, Pk, _ = np.loadtxt(spectrum, skiprows=1).T
            Pk *= 2 * pi**2 * 4 * pi
        elif k is None or Pk is None:
            raise ValueError("k and Pk must be supplied together")
        k = np.ascontiguousarray(k, dtype=np.float64)
        pk = np.ascontiguousarray(Pk, dtype=np.float64)
        if k.ndim != 1 or pk.ndim != 1 or k.size != pk.size or k.size < 2:
            raise ValueError(
                "k and Pk must be one-dimensional arrays of equal length >= 2"
            )
        if not np.all(np.isfinite(k)) or not np.all(np.isfinite(pk)):
            raise ValueError("k and Pk must contain finite values")
        if np.any(k <= 0) or np.any(np.diff(k) <= 0) or np.any(pk < 0):
            raise ValueError("k must be positive and increasing, and Pk nonnegative")
        self.k, self.Pk = k, pk
        self.k2Pk = k**2 * pk
        self.Nk = len(k)

    def Pk_CDM(self, k):
        return _power_spectrum(k, self.k, self.Pk)

    def sigma(self, i, R):
        return _sigma(i, R, self.k, self.Pk)

    def integrand(self, k, phi, theta, ikx, iky, ikz, ikk, dx, dy, dz, R1, R2):
        return _integrand(
            k, phi, theta, ikx, iky, ikz, ikk, dx, dy, dz, R1, R2, self.k, self.Pk
        )

    def integrand_lambdaCDM(self, phi, theta, ikx, iky, ikz, ikk, dx, dy, dz, R1, R2):
        return _integrand_radial(
            phi, theta, ikx, iky, ikz, ikk, dx, dy, dz, R1, R2, self.k, self.k2Pk
        )
