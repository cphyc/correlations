from numba import vectorize, guvectorize, jit
import numpy as np


@jit(['float64(float64[:], float64[:])'], nogil=True)
def trapz(A, x):
    dx = np.diff(x)
    AA = (A[..., 1:] + A[..., :-1]) / 2
    return np.sum(dx * AA, axis=-1)


@vectorize(["float64(float64)"], target='parallel')
def W1TH(x):
    x2 = x**2
    if x < 1e-4:
        return 1 - x2/10
    else:
        return 3. * (np.sin(x)/x2/x - np.cos(x)/x2)


def W1G(x):
    return np.exp(-x**2/2)


@guvectorize(
    ["void(float64[:, :], int32[:], int32[:], int32[:], int32[:], int32[:], int32[:])"],
    "(N, M)->(N),(N),(N),(N),(N),(N)",
    target='parallel')
def get_maxis(A, max0, max1, max2, v0, v1, v2):
    N, M = A.shape
    J = np.arange(0, M)
    I = np.mod(J-1, M)
    II = np.mod(J-2, M)
    K = np.mod(J+1, M)
    KK = np.mod(J+2, M)

    for i in range(N):
        AA = A[i]
        # Get local maximum
        mask = (AA[J] > AA[I]) & (AA[J] > AA[K]) & \
               (AA[I] > AA[II]) & (AA[K] > AA[KK])

        # Get values
        mvals = AA[mask]

        if len(mvals) < 3:
            max0[i] = -1
            max1[i] = -1
            max2[i] = -1
            continue

        # Order array (in decreasing order)
        order = np.argsort(-mvals)

        # Get the 3 predominant maxima
        for k in range(len(order)):
            oi = order[k]
            if oi == 0:
                i0 = k
            elif oi == 1:
                i1 = k
            elif oi == 2:
                i2 = k

        # Convert in initial coordinates
        j = 0
        ii0 = ii1 = ii2 = -1

        for k in range(M):
            if mask[k]:
                if j == i0:
                    ii0 = k
                elif j == i1:
                    ii1 = k
                elif j == i2:
                    ii2 = k
                j += 1

        max0[i] = ii0
        max1[i] = ii1
        max2[i] = ii2
        v0[i] = AA[ii0]
        v1[i] = AA[ii1]
        v2[i] = AA[ii2]


def constrain(mean, cov, values):
    '''Return the constrained mean and covariance given the values.
    values is an array of same length as mean, with np.nan where you don't want
    any constrain and the value elsewhere.

    parameters:
    * mean, (N, ) the unconstrained mean
    * cov, (N, N,) the unconstrained covariance matrix
    * values, (N, ) the constrain, containing n non-nan values

    returns:
    * (N-n, ) the constrained mean
    * (N-n, N-n) the constrained variance'''
    cons = np.isfinite(values)
    keep = np.logical_not(cons)

    # Keep only finite values (other values: no constrain)
    vals = values[cons]

    Sigma11 = cov[keep][:, keep]
    Sigma12 = cov[keep][:, cons]
    Sigma21 = cov[cons][:, keep]
    Sigma22 = cov[cons][:, cons]
    iSigma22 = np.linalg.inv(Sigma22)

    mu1 = mean[keep]
    mu2 = mean[cons]
    mean_cons = mu1 + np.dot(np.dot(Sigma12, iSigma22),
                             (vals - mu2))
    cov_cons = Sigma11 - np.dot(np.dot(Sigma12, iSigma22), Sigma21)

    return np.array(mean_cons).flatten(), np.array(cov_cons)
