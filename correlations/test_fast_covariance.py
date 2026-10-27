"""Compare the accelerated backend against the original adaptive quadrature."""

from math import prod

import numpy as np
import pytest
from numba import get_num_threads
from numpy.testing import assert_allclose
from scipy.special import spherical_jn

from correlations import Correlator, Pk, k
from correlations.correlations import _correlation, sigma
from correlations.fields import derivative_components


@pytest.mark.parametrize("order", range(9))
def test_tensor_components(order):
    components = derivative_components(order)
    assert len(components) == (order + 1) * (order + 2) // 2
    assert len(set(components)) == len(components)
    assert all(sum(p) == order for p in components)


@pytest.mark.parametrize(
    "old,new,sign",
    [
        ("potential", "phi", 1),
        ("a", "phi_i", -1),
        ("tide", "phi_ij", 1),
        ("delta", "nu", 1),
        ("grad_delta", "nu_i", 1),
        ("hessian", "nu_ij", 1),
        ("third", "nu_ijk", 1),
    ],
)
def test_legacy_aliases(old, new, sign):
    a, b = Correlator(), Correlator()
    a.add_point([0, 0, 0], [old], 1)
    b.add_point([0, 0, 0], [new], 1)
    for attr in ["kxfactor", "kyfactor", "kzfactor", "kfactor"]:
        assert getattr(a, attr) == getattr(b, attr)
    assert_allclose(a.signs, np.asarray(b.signs) * sign)


@pytest.mark.parametrize(
    "displacement", [[0, 0, 0], [1.1, -0.7, 0.4], [0, 1.5, 0], [1e-12, -2e-12, 3e-12]]
)
def test_matrix_against_reference(displacement):
    c = Correlator(backend="reference")
    c.add_point([0, 0, 0], ["phi", "a", "nu", "nu_i"], 1.2)
    c.add_point(displacement, ["phi", "a", "nu", "nu_i"], 2.1)
    expected = c.cov.copy()
    actual = c.compute_covariance(backend="numba")
    assert_allclose(actual, expected, rtol=2e-5, atol=2e-8)
    assert_allclose(actual, actual.T, atol=0, rtol=0)
    assert np.linalg.eigvalsh(actual).min() > -1e-12


@pytest.mark.parametrize("family", ["phi", "nu"])
@pytest.mark.parametrize("order", [3, 4, 5, 6])
def test_high_derivatives_at_zero(family, order):
    c = Correlator()
    field = family + "_" + "ijklmnopqrstuvwxyz"[:order]
    c.add_point([0, 0, 0], field, 1.3)
    covariance = c.cov

    # Isotropic even moments: <n_x^(2a)n_y^(2b)n_z^(2c)>.
    def odd_factorial(n):
        return prod(range(1, n + 1, 2))

    powers = derivative_components(order)
    expected = np.zeros_like(covariance)
    for i, first in enumerate(powers):
        for j, second in enumerate(powers):
            total = np.asarray(first) + second
            if np.all(total % 2 == 0):
                expected[i, j] = prod(
                    odd_factorial(int(n) - 1) for n in total
                ) / odd_factorial(2 * order + 1)
    assert_allclose(covariance, expected, rtol=3e-13, atol=1e-15)


# Select different Cartesian multiplicities, derivative parities and spectra.
@pytest.mark.parametrize(
    "first,second,i,j",
    [
        ("phi_ijk", "nu", 0, 0),
        ("nu_ijk", "phi_i", 3, 1),
        ("phi_ijkl", "nu_i", 9, 2),
        ("nu_ijkl", "phi_ij", 4, 5),
        ("phi_ijklm", "phi_ijk", 14, 8),
        ("nu_ijklm", "nu_ij", 6, 3),
        ("phi_ijklmn", "nu_ijkl", 17, 10),
        ("nu_ijklm", "nu_ijklm", 20, 19),
    ],
)
def test_high_derivative_nonzero_reference(first, second, i, j):
    c = Correlator()
    c.add_point([0, 0, 0], first, 1.2)
    c.add_point([1.7, -0.8, 1.1], second, 1.8)
    index = len(derivative_components(len(first.split("_")[1]))) + j
    a = np.array([c.kxfactor[i], c.kyfactor[i], c.kzfactor[i]])
    b = np.array([c.kxfactor[index], c.kyfactor[index], c.kzfactor[index]])
    inverse = c.kfactor[i] + c.kfactor[index]
    expected = _correlation.func(
        *map(int, a + b),
        inverse,
        1.7,
        -0.8,
        1.1,
        1.2,
        1.8,
        c.signs[i],
        c.signs[index],
        int(sum(a) - c.kfactor[i]),
        int(sum(b) - c.kfactor[index]),
    ) * (-1) ** int(sum(b))
    assert_allclose(c.cov[i, index], expected, rtol=3e-4, atol=2e-8)


def test_parallel_matches_serial_and_restores_threads():
    before = get_num_threads()
    c = Correlator()
    for p in [[0, 0, 0], [0.7, 1.1, -0.3]]:
        c.add_point(p, ["phi_ij", "nu_ijk"], 1)
    serial = c.cov.copy()
    c.nproc = min(2, before)
    parallel = c.compute_covariance()
    assert_allclose(serial, parallel, rtol=0, atol=0)
    assert get_num_threads() == before


def test_new_field_normalizations():
    c = Correlator()
    c.add_point([0, 0, 0], ["nu_i", "nu", "phi_i", "phi"], 1)
    # Relative normalizations of scalar and vector fields at finite separation.
    c.add_point([1, 0, 0], "nu", 2)
    for index, power, inverse in [(0, 1, 0), (4, 1, 2)]:
        expected = _correlation.func(
            1, 0, 0, inverse, 1, 0, 0, 1, 2, 1, 1, power - inverse, 0
        )
        assert_allclose(c.cov[index, -1], expected, rtol=1e-5, atol=1e-10)
    assert sigma(1, 1) != sigma(0, 1)


@pytest.mark.parametrize("backend", ["numba", "reference"])
def test_empty_matrix(backend):
    assert Correlator(backend=backend).cov.shape == (0, 0)


@pytest.mark.parametrize(
    "name", ["phi_", "nu_ji", "nu_x", "phi_ii", "phii", "nu_ij_extra"]
)
def test_invalid_generic_names(name):
    c = Correlator()
    with pytest.raises(ValueError):
        c.add_point([0, 0, 0], name, 1)
    assert c.Npts == 0


def test_generic_constraints():
    c = Correlator()
    values = np.full(21, np.nan)
    values[0] = 1
    c.add_point([0, 0, 0], "phi_ijklm", 1, constrains={"phi_ijklm": values}, name="A")
    assert np.array_equal(c["A"]["phi_ijklm"], np.arange(21))
    assert c.c.mean.shape == (20,)
    assert c.c.cov.shape == (20, 20)


@pytest.mark.parametrize("distance", [1e-50, 1e-9, 1, 1000])
def test_scalar_density_radial_integral(distance):
    c = Correlator()
    c.add_point([0, 0, 0], "nu", 1.5)
    c.add_point([distance, 0, 0], "nu", 2)
    expected = np.trapezoid(
        k**2
        * Pk
        * np.exp(-0.5 * k**2 * (1.5**2 + 2**2))
        * spherical_jn(0, k * distance),
        k,
    )
    expected /= 2 * np.pi**2 * sigma(0, 1.5) * sigma(0, 2)
    assert_allclose(c.cov[0, 1], expected, rtol=2e-13, atol=1e-15)


def test_high_derivative_near_zero_separation():
    c = Correlator()
    c.add_point([0, 0, 0], "nu_ijklm", 1)
    c.add_point([1e-14, -2e-14, 3e-14], "nu_ijklm", 1)
    covariance = c.cov
    assert_allclose(covariance[:21, 21:], covariance[:21, :21], rtol=2e-13, atol=1e-15)


def test_fully_constrained_and_partial_symmetries():
    c = Correlator()
    c.add_point([0, 0, 0], "nu_i", 1, constrains={"nu_i": [1, 2, 3]})
    c.add_point([1, 0, 0], "nu", 1)
    assert c.c.cov.shape == (1, 1)
    assert c.c.mean.shape == (1,)
    assert c.cov[1, 3] == c.cov[2, 3] == 0


def test_rotated_tensors_share_radial_basis(monkeypatch):
    from correlations import fast_covariance as fast

    positions = np.array([[0, 0, 0], [3, 4, 0], [0, 3, 4], [4, 0, 3]])
    c = Correlator()
    fields = ["phi", "phi_i", "phi_ij", "phi_ijk"]
    for position in positions:
        c.add_point(position, fields, 1)
    original = fast._bessel_table
    calls = []

    def counted(order, z):
        calls.append((order, z.copy()))
        return original(order, z)

    monkeypatch.setattr(fast, "_bessel_table", counted)
    covariance = c.cov
    distances = {np.linalg.norm(b - a) for a in positions for b in positions}
    assert len(calls) == len(distances)
    assert all(order == 6 for order, _ in calls)  # Seven moments per distance.

    # Cyclic rotation (x,y,z)->(z,x,y), checked across every rank 0–3.
    permutation = []
    offset = 0
    for order in range(4):
        components = derivative_components(order)
        permutation.extend(
            offset + components.index((p[1], p[2], p[0])) for p in components
        )
        offset += len(components)
    original_block = covariance[:20, 20:40]
    rotated_block = covariance[:20, 40:60]
    assert_allclose(
        rotated_block,
        original_block[np.ix_(permutation, permutation)],
        rtol=3e-13,
        atol=2e-15,
    )


def test_mixed_field_blocks_and_normalizations():
    fields = ["phi", "phi_i", "phi_ij", "nu", "nu_i", "nu_ij", "a"]
    points = [([0, 0, 0], 1), ([1, 0, 0], 2), ([0, -1, 0], 1.5)]
    combined = Correlator()
    for position, radius in points:
        combined.add_point(position, fields, radius)
    actual = combined.cov
    width = 23
    for i, (position, radius) in enumerate(points):
        for j in range(i, len(points)):
            other, other_radius = points[j]
            pair = Correlator(backend="reference")
            pair.add_point(position, fields, radius)
            pair.add_point(other, fields, other_radius)
            expected = pair.cov[:width, width:]
            assert_allclose(
                actual[i * width : (i + 1) * width, j * width : (j + 1) * width],
                expected,
                rtol=3e-5,
                atol=2e-8,
            )


def test_planning_keeps_distinct_float_separations():
    from correlations.fast_covariance import _plan_tasks

    positions = np.array([[0, 0, 0], [1, 0, 0], [np.nextafter(1.0, 2.0), 0, 0]])
    _, geometries, _ = _plan_tasks(
        positions,
        np.ones(3),
        np.zeros((3, 3), dtype=np.int64),
        np.zeros(3, dtype=np.int64),
    )
    displacements = {g[0] for g in geometries}
    assert 1.0 in displacements
    assert np.nextafter(1.0, 2.0) in displacements
