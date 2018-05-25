from correlations.utils import Utils
from correlations.correlations import integrand_python
import numpy as np
from time import time
from scipy.integrate import tplquad, dblquad, nquad

k, Pk, blip = np.loadtxt('/home/ccc/Documents/prog/correlations/correlations/data/power.dat', skiprows=1).T
Pk *= 8*np.pi**3

u = Utils(k, Pk)


def check_close(a, b, rtol=1e-4):
    if not np.isclose(a, b, rtol=rtol):
        print('%s != %s' % (a, b))
        assert False


def test_Pk():
    """Test that the cython-version of Pk recovers Pk"""
    assert np.allclose(Pk, np.vectorize(u.Pk_CDM)(k))


def test_sigma():
    def ref_sigma(i, R):
        integrand = k**(2*i+2) * Pk * np.exp(-(k * R)**2) / (2*np.pi**2)
        ret = np.sqrt(np.sum((integrand[1:] + integrand[:-1]) * np.diff(k)) / 2)
        return ret

    print(ref_sigma(0, 8))
    for i in range(-2, 3):
        for R in np.linspace(1e-1, 20):
            print(i, R)
            check_close(ref_sigma(i, R), u.sigma(i, R))


def test_integrand():
    """Test the integrand is the same over a sample of theta and phi"""
    args = 0, 0, 0, 0, 0, 0, 0, 1, 1

    for i in range(1000):
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * 2 * np.pi

        def check_it(theta, phi):

            intgd = np.vectorize(lambda kk: u.integrand(kk, phi, theta, *args))(k)

            a = integrand_python(phi, theta, 0, 0, 0, 0, 0, 0, 0, 1, 1)
            b = np.trapz(intgd, k)
            c = u.integrand_lambdaCDM(phi, theta, *args)

            print(a, b, c)

            assert np.isclose(a, b, rtol=1e-3)
            assert np.isclose(a, c, rtol=1e-3)

        check_it(theta, phi)


def test_integration():
    kmin, kmax = k[0], k[-1]
    ikx, iky, ikz, ikk = 0, 0, 0, 0
    xyz = np.array([0, 0, 0])

    def test_x(x):
        xyz[0] = x

        ref = dblquad(integrand_python,
                      0, np.pi,
                      lambda _: 0, lambda _: 2*np.pi,
                      args=(ikx, iky, ikz, ikk, *xyz, 1, 1))[0]

        def do_integration(x, integrator, integrand):
            a, da = integrator(integrand, *bounds, **kwa)
            print('expected %s, got %s' % (ref, a))
            assert np.isclose(a, ref, rtol=1e-3)

        kwa = dict(epsrel=1e-5,
                   args=(ikx, iky, ikz, ikk, *list(xyz), 1, 1))

        bounds = (0, np.pi, lambda _: 0, lambda _: 2*np.pi)
        yield do_integration, x, dblquad, u.integrand_lambdaCDM

        bounds = (0, np.pi, lambda _: 0, lambda _: 2*np.pi, lambda _1, _2: 0, lambda _1, _2: np.inf)
        yield do_integration, x, tplquad, u.integrand

        kwa = dict(
            opts=[
                {'epsrel': 1e-6},  # k integral
                {'epsrel': 1e-6},  # phi integral
                {'epsrel': 1e-6}   # theta integral
            ],
            args=(ikx, iky, ikz, ikk, *list(xyz), 1, 1))

        # Note: the integration order is the opposite for nquad...
        bounds = ([(0, np.inf), (0, 2*np.pi), (0, np.pi)], )
        yield do_integration, x, nquad, u.integrand

    for x in np.linspace(0, 2, 20):
        for _ in test_x(x):
            yield _
