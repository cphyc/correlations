"""Exact angular integration and Numba-compiled radial covariance sums.

For a Cartesian monomial n_x**a n_y**b n_z**c, angular integration of
i**(a+b+c) exp(-i k n.d) is a finite contraction sum. Pairing p_x, p_y,
p_z indices gives coefficient prod(a! / ((a-2p)! p! 2**p)) and radial
factor (-1)**p j_(m-p)(kr)/(kr)**p, where m=a+b+c and p=sum(p_i).
Unpaired indices supply powers of the unit separation vector. The Fourier
inverse-k factors and the derivative parity at the second point supply
the remaining signs, exactly as in the original dblquad implementation.

Writing q=m-p, the radial term times its unpaired direction factors is
prod(d_axis**unpaired) * H_q, with
H_q = integral dk k**(2+2q-inverse) P(k) W1 W2 j_q(kr)/(kr)**q / (2*pi**2).
These scalar moments depend only on distance, smoothing, and inverse-k power.
All tensor ranks through a maximum combined order M share just M+1 moments;
rotations change only the Cartesian monomials, not the integrals. Scaled
Bessel functions retain the analytic limits at zero separation.

Only the angular integral changes: the k grid, smoothing, normalization,
and trapezoidal weights are identical to the reference backend. No angular
resolution parameter, fast-math approximation, or disk result cache is used.
"""

from functools import lru_cache
from itertools import product
from math import factorial

import numpy as np
from numba import config, get_num_threads, njit, prange, set_num_threads
from scipy.special import spherical_jn


@lru_cache(None)
def _contractions(powers):
    terms = []
    for pairs in product(*(range(n // 2 + 1) for n in powers)):
        p = sum(pairs)
        coefficient = (-1.0) ** p
        unpaired = []
        for n, count in zip(powers, pairs, strict=True):
            left = n - 2 * count
            coefficient *= factorial(n) / (
                factorial(left) * factorial(count) * 2**count
            )
            unpaired.append(left)
        terms.append((sum(powers) - p, p, coefficient, tuple(unpaired)))
    return terms


@njit(cache=True, nogil=True)
def _small_bessel_table(order, z):
    """Stable j_l(z)/z**l at and near zero, including its analytic limit."""
    table = np.empty((order + 1, len(z)))
    odd_factorial = 1.0
    for ell in range(order + 1):
        odd_factorial *= 2 * ell + 1
        for s in range(len(z)):
            term = 1 / odd_factorial
            value = term
            for j in range(1, 17):
                term *= -(z[s] ** 2) / (2 * j * (2 * ell + 2 * j + 1))
                value += term
            table[ell, s] = value
    return table


def _bessel_table(order, z):
    """Scaled spherical Bessel functions j_l(z)/z**l for all radial orders."""
    small = z < 1
    table = np.empty((order + 1, len(z)))
    ell = np.arange(order + 1)[:, None]
    large_z = z[~small][None, :]
    table[:, ~small] = spherical_jn(ell, large_z) / large_z**ell
    table[:, small] = _small_bessel_table(order, z[small])
    return table


@njit(cache=True, nogil=True)
def _radial_moment(ell, inverse, k, bessel, weight):
    result = 0.0
    for s in range(len(k)):
        result += weight[s] * k[s] ** (2 * ell - inverse) * bessel[ell, s]
    return result


@njit(cache=True, nogil=True)
def _radial_serial(k, bessel, weight, inverse):
    values = np.empty(len(bessel))
    for ell in range(len(values)):
        values[ell] = _radial_moment(ell, inverse, k, bessel, weight)
    return values


@njit(cache=True, nogil=True, parallel=True)
def _radial_parallel(k, bessel, weight, inverse):
    values = np.empty(len(bessel))
    for ell in prange(len(values)):
        values[ell] = _radial_moment(ell, inverse, k, bessel, weight)
    return values


@njit(cache=True, nogil=True)
def _contract_tasks(
    tasks,
    geometries,
    radial_ids,
    basis,
    power_ids,
    offsets,
    ell,
    unpaired,
    coefficients,
):
    """Reconstruct Cartesian tensor components from shared scalar moments."""
    values = np.empty(len(tasks))
    for task in range(len(tasks)):
        d = geometries[tasks[task, 0]]
        power = power_ids[task]
        value = 0.0
        for term in range(offsets[power], offsets[power + 1]):
            value += (
                coefficients[term]
                * d[0] ** unpaired[term, 0]
                * d[1] ** unpaired[term, 1]
                * d[2] ** unpaired[term, 2]
                * basis[radial_ids[task], ell[term]]
            )
        values[task] = value
    return values


@njit(cache=True, nogil=True)
def _plan_tasks(positions, scales, powers, inverse):
    """Map upper-triangular entries to distinct, unnormalized radial integrals.

    Geometry keys use the exact input floats. Fields with different signs or
    spectral normalizations can share an integral; those factors are applied
    only when assembling the final matrix.
    """
    size = len(inverse)
    pair_tasks = np.full(size * (size + 1) // 2, -1, dtype=np.int64)
    geometry_ids = {(0.0, 0.0, 0.0, 0.0): -1}
    geometry_ids.pop((0.0, 0.0, 0.0, 0.0))
    task_ids = {(0, 0, 0, 0, 0): -1}
    task_ids.pop((0, 0, 0, 0, 0))
    geometries = []
    tasks = []
    entry = 0
    for i in range(size):
        for j in range(i, size):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            a = powers[i, 0] + powers[j, 0]
            b = powers[i, 1] + powers[j, 1]
            c = powers[i, 2] + powers[j, 2]
            if (dx == 0 and a % 2) or (dy == 0 and b % 2) or (dz == 0 and c % 2):
                entry += 1
                continue
            geometry = (dx, dy, dz, scales[i] ** 2 + scales[j] ** 2)
            group = geometry_ids.get(geometry, -1)
            if group == -1:
                group = len(geometries)
                geometry_ids[geometry] = group
                geometries.append(geometry)
            key = (group, a, b, c, inverse[i] + inverse[j])
            task = task_ids.get(key, -1)
            if task == -1:
                task = len(tasks)
                task_ids[key] = task
                tasks.append(key)
            pair_tasks[entry] = task
            entry += 1
    return pair_tasks, geometries, tasks


@njit(cache=True, nogil=True)
def _assemble_covariance(pair_tasks, values, powers, inverse, signs, norms):
    size = len(norms)
    covariance = np.zeros((size, size))
    entry = 0
    for i in range(size):
        for j in range(i, size):
            task = pair_tasks[entry]
            if task != -1:
                parity = (
                    powers[j, 0]
                    + powers[j, 1]
                    + powers[j, 2]
                    + (inverse[i] + inverse[j]) // 2
                )
                value = values[task] * (-1.0) ** parity
                value *= signs[i] * signs[j] / (norms[i] * norms[j])
                covariance[i, j] = value
                covariance[j, i] = value
            entry += 1
    return covariance


def compute_covariance(correlator, *, nthreads=1):
    """Compute each distinct integral once, then assemble the symmetric matrix.

    Temporary Bessel storage is O(max derivative order * Nk), independent
    of matrix size. A packed upper-triangular integer map associates entries
    with integral tasks. nthreads masks Numba's worker pool for this call and
    restores the caller's thread setting afterward.
    """
    if nthreads > config.NUMBA_NUM_THREADS:
        raise ValueError(
            f"nproc exceeds Numba's thread limit ({config.NUMBA_NUM_THREADS})"
        )
    k, pk = np.asarray(correlator.k), np.asarray(correlator.Pk)
    powers = np.array(
        [correlator.kxfactor, correlator.kyfactor, correlator.kzfactor], dtype=np.int64
    ).T
    inverse = np.asarray(correlator.kfactor, dtype=np.int64)
    positions = correlator.positions
    scales = correlator.smoothing_scales
    signs = np.asarray(correlator.signs)
    size = len(inverse)
    if size == 0:
        return np.zeros((0, 0))

    from .correlations import sigma

    kinds = {
        (int(sum(p) - inv), r)
        for p, inv, r in zip(powers, inverse, scales, strict=True)
    }
    norms_by_kind = {kind: sigma(*kind) for kind in kinds}
    norms = np.array(
        [
            norms_by_kind[int(sum(p) - inv), r]
            for p, inv, r in zip(powers, inverse, scales, strict=True)
        ]
    )
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Spectral moments must be finite and positive")

    # Endpoint half-weights reproduce the reference's radial trapezoidal sum.
    widths = np.diff(k)
    weights = np.empty_like(k)
    weights[0], weights[-1] = widths[0] / 2, widths[-1] / 2
    weights[1:-1] = (widths[:-1] + widths[1:]) / 2
    weights *= k**2 * pk / (2 * np.pi**2)

    pair_tasks, geometries, tasks = _plan_tasks(positions, scales, powers, inverse)
    tasks = np.asarray(tasks, dtype=np.int64)
    geometries = np.asarray(geometries)
    radii = np.linalg.norm(geometries[:, :3], axis=1)
    radial_keys, radial_orders, radial_ids = (
        {},
        [],
        np.empty(len(tasks), dtype=np.int64),
    )
    # A radial basis depends on distance and the sum of squared smoothing
    # scales, not the orientation or individual Cartesian components.
    for index, (geometry, a, b, c, inv) in enumerate(tasks):
        key = (radii[geometry], geometries[geometry, 3], inv)
        group = radial_keys.get(key)
        if group is None:
            group = len(radial_keys)
            radial_keys[key] = group
            radial_orders.append(0)
        radial_orders[group] = max(radial_orders[group], a + b + c)
        radial_ids[index] = group
    basis = np.zeros((len(radial_keys), int(tasks[:, 1:4].sum(axis=1).max()) + 1))

    # Contraction coefficients depend only on derivative multiplicities.
    # Direction enters later as Cartesian monomials in the compiled loop.
    unique_powers, power_ids = np.unique(tasks[:, 1:4], axis=0, return_inverse=True)
    offsets, ell, unpaired, coefficients = [0], [], [], []
    for powers_tuple in unique_powers:
        for q, _, coefficient, left in _contractions(tuple(powers_tuple)):
            ell.append(q)
            unpaired.append(left)
            coefficients.append(coefficient)
        offsets.append(len(ell))

    old_threads = get_num_threads()
    try:
        if nthreads > 1:
            set_num_threads(nthreads)
        integrate = _radial_parallel if nthreads > 1 else _radial_serial
        for (r, scale2, inv), group in correlator._pbar(
            radial_keys.items(), total=len(radial_keys)
        ):
            bessel = _bessel_table(radial_orders[group], k * r)
            weight = weights * np.exp(-0.5 * k**2 * scale2)
            basis[group, : len(bessel)] = integrate(k, bessel, weight, inv)
    finally:
        if nthreads > 1:
            set_num_threads(old_threads)
    values = _contract_tasks(
        tasks,
        geometries,
        radial_ids,
        basis,
        power_ids,
        np.asarray(offsets, dtype=np.int64),
        np.asarray(ell, dtype=np.int64),
        np.asarray(unpaired, dtype=np.int64),
        np.asarray(coefficients),
    )
    return _assemble_covariance(pair_tasks, values, powers, inverse, signs, norms)
