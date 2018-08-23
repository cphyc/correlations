from correlations.correlations import Correlator, sigma
import numpy as np
from pprint import pprint

def is_def_positive(A):
    eigvals = np.linalg.eigvalsh(A)
    if not all(eigvals > 0):
        print('Matrix not inversible', eigvals)
    assert all(eigvals > 0)


def test_validation_invalidation():
    '''Validation/invalidation'''
    c = Correlator(quiet=True)
    c.add_point([0, 0, 0], ['delta'], 1)

    # Compute covariance
    c.compute_covariance()

    # Check shape
    cov = c.cov
    assert cov.shape == (1, 1)

    # Check invalidation
    assert c._covariance_valid is True
    c.add_point([1, 0, 0], ['delta'], 1)

    assert c._covariance_valid is False

    # Automatic computation
    cov = c.cov
    assert cov.shape == (2, 2)
    assert c._covariance_valid is True

    # Check the matrix is definite positive
    is_def_positive(c.cov)


def test_nu():
    '''nu-nu correlations'''
    c = Correlator(quiet=True)

    for x in np.linspace(0, 10, 10):
        c.add_point([x, 0, 0], ['delta'], 1)

    # Check unit variance on diagonal
    assert np.allclose(np.diag(c.cov), 1)

    # Check the matrix is definite positive
    is_def_positive(c.cov)


def test_zerolag():
    '''0 lag covariance'''
    c = Correlator(quiet=True)
    c.add_point([0, 0, 0],
                ['potential', 'a', 'tide', 'grad_delta', 'hessian'],
                1)

    # Kronecker symbol
    def K(i, j):
        return 1 if i == j else 0

    # Precompute sigmas
    sigmam2 = sigma(-2, 1)
    sigmam1 = sigma(-1, 1)
    sigma0 = sigma(0, 1)
    sigma1 = sigma(1, 1)
    sigma2 = sigma(2, 1)

    # Build expected answer
    expected = np.zeros((19, 19))
    expected[0, 0] = 1
    expected[1:4, 1:4] = np.eye(1)/3

    # Hessians
    indexes = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    for i0, (i, j) in enumerate(indexes):
        for i1, (k, l) in enumerate(indexes):
            tmp = (K(i, j)*K(k, l) + K(i, k)*K(j, l) + K(i, l)*K(j, k)) / 15
            expected[4+i0, 4+i1]   = tmp
            expected[13+i0, 13+i1] = tmp
            expected[4+i0, 13+i1]  = -tmp * sigma1**2 / sigma0 / sigma2

    # Gradients
    expected[1:4, 1:4] = expected[10:13, 10:13] = np.eye(3) / 3

    # phi-tide
    gamma = (sigmam1**2 / sigmam2 / sigma0)
    expected[0, 4:7] = -gamma / 3

    # phi-hessian
    gamma = sigma0**2 / sigmam2 / sigma2
    expected[0, 13:16] = gamma / 3

    # grad phi - grad nu
    expected[1:4, 10:13] = np.eye(3) / 3 * sigma0**2 / sigma1 / sigmam1

    # Symmetrize expected matrix
    expected = np.triu(expected)
    expected = (expected + expected.T) - np.diag(np.diag(expected))

    pprint(np.round(c.cov, 3)[:10, :10])
    pprint(np.round(expected, 3)[:10, :10])
    print((c.cov - expected) / expected)
    # print(np.round(c.cov*15, 5))
    # print(np.round(expected*15, 5))
    assert np.allclose(c.cov, expected)

    # Check the matrix is definite positive
    is_def_positive(c.cov)


def test_gradgrad():
    '''grad-grad correlations'''
    c = Correlator(quiet=True)
    c.add_point([0, 0, 0], ['grad_delta'], 1)
    c.add_point([1, 0, 0], ['grad_delta'], 1)

    print(c.cov)

    # Perpendicular gradients do not correlate,
    # parallel gradients correlate positively
    for i in range(3):
        for j in range(3):
            if i == j:
                assert c.cov[i, 3+j] > 0
            else:
                assert np.isclose(c.cov[i, 3+j], 0)

    is_def_positive(c.cov)
