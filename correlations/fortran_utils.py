from ctypes import POINTER, c_double, c_int
from os.path import split, abspath, join

import numpy as np
from ctypes import CDLL

_dir = split(__file__)[0]


def compute_covariance(
    k, Pk, X, R, ikx, iky, ikz, ikk, signs, epsrel=1e-5, epsabs=1e-7
):
    fortran = CDLL(abspath(join(_dir, "_fortran_utils.so")))
    k = np.require(k, np.float64, ["C_CONTIGUOUS"])
    Pk = np.require(Pk, np.float64, ["C_CONTIGUOUS"])
    x, y, z = (
        np.require(X[:, idim], np.float64, ["C_CONTIGUOUS"]) for idim in range(3)
    )

    R = np.require(R, np.float64, ["C_CONTIGUOUS"])
    ikx = np.require(ikx, np.int32, ["C_CONTIGUOUS"])
    iky = np.require(iky, np.int32, ["C_CONTIGUOUS"])
    ikz = np.require(ikz, np.int32, ["C_CONTIGUOUS"])
    ikk = np.require(ikk, np.int32, ["C_CONTIGUOUS"])
    signs = np.require(signs, np.int32, ["C_CONTIGUOUS"])

    npt = len(x)

    covariance = np.empty((npt, npt), dtype=np.float64)

    def i2ptr(A):
        return A.ctypes.data_as(POINTER(c_int))

    def d2ptr(A):
        return A.ctypes.data_as(POINTER(c_double))

    Nk = len(Pk)

    print()

    fortran.init(d2ptr(k), d2ptr(Pk), c_int(Nk), c_double(epsrel), c_double(epsabs))
    fortran.compute_covariance(
        d2ptr(x),
        d2ptr(y),
        d2ptr(z),
        d2ptr(R),
        i2ptr(ikx),
        i2ptr(iky),
        i2ptr(ikz),
        i2ptr(ikk),
        i2ptr(signs),
        d2ptr(covariance),
        c_int(npt),
    )

    return covariance
