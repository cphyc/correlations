from correlations.utils import sigma_TH, Pk_CDM, compute_correlation, integrand, integrand_lambdaCDM
from correlations.correlations import _correlation, integrand as py_integrand
import numpy as np
from time import time
from scipy.integrate import tplquad, dblquad, nquad

k, Pk, blip = np.loadtxt('/home/ccc/Documents/prog/correlations/correlations/data/power.dat', skiprows=1).T
Pk *= 8*np.pi**3


def test_Pk():
    """Test that the cython-version of Pk recovers Pk"""
    assert np.allclose(Pk, np.vectorize(Pk_CDM)(k))
    assert np.isclose(sigma_TH(0, 8), 0.81590744, rtol=1e-4)


ikx, iky, ikz, ikk = 1, 0, 0, 2
dx, dy, dz = 0, 0, 0
R1, R2 = 1, 1
sign1, sign2 = 1, 1
nsigma1, nsigma2 = 0, 0


def test_integrand():
    """Test the integrand is the same over a sample of theta and phi"""
    args = 0, 0, 0, 0, 0, 0, 0, 1, 1

    for i in range(1000):
        theta = np.random.rand() * np.pi
        phi = np.random.rand() * 2 * np.pi

        def check_it(theta, phi):

            intgd = np.vectorize(lambda kk: integrand(kk, phi, theta, *args))(k)

            a = py_integrand(phi, theta, 0, 0, 0, 0, np.array([0, 0, 0]), 1, 1)
            b = np.trapz(intgd, k)
            c = integrand_lambdaCDM(phi, theta, *args)

            print(a, b, c)

            assert np.isclose(a, b, rtol=1e-3)
            assert np.isclose(a, c, rtol=1e-3)

        check_it(theta, phi)


def test_integration():
    """Integration should converge to the same result"""
    for dx in np.linspace(0, 10):
        before = time()
        a = compute_correlation(ikx, iky, ikz, ikk, dx, dy, dz, R1, R2,
                                sign1, sign2, nsigma1, nsigma2,
                                Pkchoice='power-law')
        print('CDM: %10.1fms' % ((time() - before)*1e3), end='\t')

        b = compute_correlation(ikx, iky, ikz, ikk, dx, dy, dz, R1, R2,
                                sign1, sign2, nsigma1, nsigma2,
                                Pkchoice='Lambda-CDM')
        print('trapz: %10.1fms' % ((time() - before)*1e3), end='\t')

        before = time()
        c = _correlation(ikx, iky, ikz, ikk, dx, dy, dz, R1, R2,
                         sign1, sign2, nsigma1, nsigma2)
        print('Numpy:  %10.1fms' % ((time() - before)*1e3))

        if not (np.isclose(a, b, rtol=1e-3) and np.isclose(a, c, rtol=1e-3)):
            print('E: %s %s %s' % (a, b, c))
            assert np.isclose(a, b, rtol=1e-3)
            assert np.isclose(a, c, rtol=1e-3)

def test_integration():
    kmin, kmax = k[0], k[-1]
    ikx, iky, ikz, ikk = 0, 0, 0, 0
    xyz = np.array([0, 0, 0])

    def test_x(x):
        ref = dblquad(py_integrand,
                      0, np.pi,
                      lambda _: 0, lambda _: 2*np.pi,
                      args=(ikx, iky, ikz, ikk, xyz, 1, 1))[0]

        def do_integration(x, integrator, integrand):
            a, da = integrator(integrand, *bounds, **kwa)
            print('expected %s, got %s' % (ref, a))
            assert np.isclose(a, ref, rtol=1e-3)

        xyz[0] = x
        kwa = dict(epsrel=1e-6,
                   args=(ikx, iky, ikz, ikk, *list(xyz), 1, 1))

        bounds = (0, np.pi, lambda _: 0, lambda _: 2*np.pi)
        yield do_integration, x, dblquad, integrand_lambdaCDM

        bounds = (0, np.pi, lambda _: 0, lambda _: 2*np.pi, lambda _1, _2: 0, lambda _1, _2: np.inf)
        yield do_integration, x, tplquad, integrand

        kwa = dict(
            opts=[
                {'epsrel': 1e-6},  # k integral
                {'epsrel': 1e-6},  # phi integral
                {'epsrel': 1e-6, 'weight': 'alg', 'wvar': (0, 0)}   # theta integral
            ],
            args=(ikx, iky, ikz, ikk, *list(xyz), 1, 1))
        
        # Note: the integration order is the opposite for nquad...
        bounds = ([(kmin, kmax), (0, 2*np.pi), (0, np.pi)], )
        yield do_integration, x, nquad, integrand

    for x in np.linspace(0, 2, 20):
        for _ in test_x(x):
            yield _
