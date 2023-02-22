from pathlib import Path

import numpy as np
from scipy.integrate import dblquad, nquad, tplquad

from correlations.correlations import integrand_python
from correlations.utils import Utils

k, Pk, blip = np.loadtxt(Path(__file__).parent / "data" / "power.dat", skiprows=1).T
Pk *= 8 * np.pi**3

u = Utils(k, Pk)


def test_Pk():
    """Test that the cython-version of Pk recovers Pk"""
    assert np.allclose(Pk, np.vectorize(u.Pk_CDM)(k))


def test_sigma():
    def ref_sigma(i, R):
        integrand = k ** (2 * i + 2) * Pk * np.exp(-((k * R) ** 2)) / (2 * np.pi**2)
        ret = np.sqrt(np.sum((integrand[1:] + integrand[:-1]) * np.diff(k)) / 2)
        return ret

    for i in range(-2, 3):
        for R in np.linspace(1e-1, 20):
            np.testing.assert_allclose(ref_sigma(i, R), u.sigma(i, R))


def test_integrand():
    """Test the integrand is the same over a sample of theta and phi"""
    args = 0, 0, 0, 0, 0, 0, 0, 1, 1

    for _ in range(1000):
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * 2 * np.pi

        def check_it(theta, phi):
            intgd = np.asarray([u.integrand(kk, phi, theta, *args) for kk in k])

            a = integrand_python(phi, theta, 0, 0, 0, 0, 0, 0, 0, 1, 1)
            b = np.trapz(intgd, k)
            c = u.integrand_lambdaCDM(phi, theta, *args)

            np.testing.assert_allclose(a, b, rtol=1e-3)
            np.testing.assert_allclose(a, c, rtol=1e-3)

        check_it(theta, phi)


def test_integration():
    ikx, iky, ikz, ikk = 0, 0, 1, 2  # This is chosen randomly
    xyz = np.array([0, 0, 0])

    def test_x(x, ikx, ikk, iky):
        xyz[0] = x

        ref = dblquad(
            integrand_python,
            0,
            np.pi,
            lambda _: 0,
            lambda _: 2 * np.pi,
            args=(ikx, iky, ikz, ikk, *xyz, 1, 1),
        )[0]

        def do_integration(x, ikx, iky, ikk, integrator, integrand):
            a, da = integrator(integrand, *bounds, **kwa)
            print(f"expected {ref}, got {a}")
            np.testing.assert_allclose(a, ref, atol=1e-10)

        kwa = {"epsrel": 1e-5, "args": (ikx, iky, ikz, ikk, *list(xyz), 1, 1)}

        bounds = (0, np.pi, lambda _: 0, lambda _: 2 * np.pi)
        do_integration(x, ikx, iky, ikk, dblquad, u.integrand_lambdaCDM)

        bounds = (
            0,
            np.pi,
            lambda _: 0,
            lambda _: 2 * np.pi,
            lambda _1, _2: 0,
            lambda _1, _2: np.inf,
        )
        do_integration(x, ikx, iky, ikk, tplquad, u.integrand)

        kwa = {
            "opts": [
                {"epsrel": 1e-6},  # k integral
                {"epsrel": 1e-6},  # phi integral
                {"epsrel": 1e-6},  # theta integral
            ],
            "args": (ikx, iky, ikz, ikk, *list(xyz), 1, 1),
        }

        # Note: the integration order is the opposite for nquad...
        bounds = ([(0, np.inf), (0, 2 * np.pi), (0, np.pi)],)
        do_integration(x, ikx, iky, ikk, nquad, u.integrand)

    for x in np.linspace(0, 2, 5):
        for ikk in [2, 0]:
            for ikx in [0, 1, 2]:
                for iky in [0, 1]:
                    test_x(x, ikx, ikk, iky)
