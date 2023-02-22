import numpy as np
from numpy.testing import assert_allclose
from scipy.special import jn, spherical_jn

from correlations import correlation_functions as cf
from correlations.correlations import Correlator, sigma


def test_helper_base():
    x = np.geomspace(1e-8, 10, 1000)
    tol = 1e-10
    assert_allclose(cf.j0(x), spherical_jn(0, x), rtol=tol)
    assert_allclose(cf.j1(x), spherical_jn(1, x), rtol=tol)
    assert_allclose(cf.j2(x), spherical_jn(2, x), rtol=tol)

    assert_allclose(cf.j1_o_x(x), spherical_jn(1, x) / x, rtol=tol)
    assert_allclose(cf.J1_o_x(x), jn(1, x) / x, rtol=tol)

    assert_allclose(cf.j2_o_x(x), spherical_jn(2, x) / x, rtol=tol)
    assert_allclose(cf.j2_o_x2(x), spherical_jn(2, x) / x**2, rtol=tol)


def test_helper_complex():
    x = np.geomspace(1e-8, 10, 1000)
    tol = 1e-10

    _jn = spherical_jn

    assert_allclose(cf.xj2_mj1_o_x(x), (x * _jn(2, x) - _jn(1, x)) / x, rtol=tol)
    assert_allclose(cf._2j2_mxj1_o_x(x), (2 * _jn(2, x) - x * _jn(1, x)) / x, rtol=tol)
    assert_allclose(
        cf._8j2_m4xj1_x2j0_o_x2(x),
        (8 * _jn(2, x) - 4 * x * _jn(1, x) + x**2 * _jn(0, x)) / x**2,
        rtol=tol,
    )


def test_Gamma_dgradd_gg():
    R1 = 1
    R2 = 2
    d = 3
    c = Correlator()
    c.add_point([0, 0, 0], ["density"], R1)
    c.add_point([d, 0, 0], ["grad_delta"], R2)

    tgt = c.cov[0, 1:] * sigma(0, R1) * sigma(1, R2)
    val = [cf.Gamma_dgradd_gg(R1, R2, x) for x in (d, 0, 0)]

    assert_allclose(tgt, val)


def test_Gamma_dhess_gg():
    R1 = 5
    R2 = 2
    dr1 = 3
    ret = np.zeros(6)
    for i, el in enumerate(
        ["r r", "theta theta", "phi phi", "r theta", "r phi", "theta phi"]
    ):
        ret[i] = cf.Gamma_dhess_gg(R1, R2, dx=dr1, element=el)

    c = Correlator()
    c.add_point([0, 0, 0], ["delta"], R1)
    c.add_point([dr1, 0, 0], ["hessian"], R2)

    ref = c.cov[0, 1:] * sigma(0, R1) * sigma(2, R2)
    assert_allclose(ref, ret)


def test_Gamma_gradg_hessg():
    R1 = 1
    R2 = 2
    dx = 3
    cov = np.zeros((3, 6))
    for i, el1 in enumerate(("r", "theta", "phi")):
        for j, el2 in enumerate(
            ("r r", "theta theta", "phi phi", "r theta", "r phi", "theta phi")
        ):
            el = f"{el1} {el2}"
            cov[i, j] = cf.Gamma_graddhess_gg(R1, R2, dx, element=el)

    c = Correlator()
    c.add_point([0, 0, 0], ["grad_delta"], R1)
    c.add_point([dx, 0, 0], ["hessian"], R2)

    tgt = c.cov[:3, 3:9] * sigma(1, R1) * sigma(2, R2)

    assert_allclose(tgt, cov)


def test_Gamma_hessg_hessg_lag():
    R1 = 1
    R2 = 2
    dx = 3
    hess_elems = ("r r", "theta theta", "phi phi", "r theta", "r phi", "theta phi")
    cov = np.zeros((6, 6))
    for i, el1 in enumerate(hess_elems):
        for j, el2 in enumerate(hess_elems):
            el = f"{el1} {el2}"
            cov[i, j] = cf.Gamma_hesshess_gg(R1, R2, dx, element=el)

    c = Correlator()
    c.add_point([0, 0, 0], ["hessian"], R1)
    c.add_point([dx, 0, 0], ["hessian"], R2)

    tgt = c.cov[:6, 6:12] * sigma(2, R1) * sigma(2, R2)

    assert_allclose(tgt, cov)


def test_Gamma_hessg_hessg_nolag():
    R1 = 1
    R2 = 2
    dx = 0
    hess_elems = ("r r", "theta theta", "phi phi", "r theta", "r phi", "theta phi")
    cov = np.zeros((6, 6))
    for i, el1 in enumerate(hess_elems):
        for j, el2 in enumerate(hess_elems):
            el = f"{el1} {el2}"
            cov[i, j] = cf.Gamma_hesshess_gg(R1, R2, dx, element=el)

    c = Correlator()
    c.add_point([0, 0, 0], ["hessian"], R1)
    c.add_point([dx, 0, 0], ["hessian"], R2)

    tgt = c.cov[:6, 6:12] * sigma(2, R1) * sigma(2, R2)

    assert_allclose(tgt, cov)
