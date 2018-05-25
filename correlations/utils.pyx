import numpy as np
cimport numpy as np
cimport cython
from scipy.integrate import quad, dblquad, tplquad
from scipy.special import spherical_jn
# from .cubature.cubature import cubature

cdef extern from "math.h":
    double sqrt(double x)
    double sin(double x)
    double cos(double x)
    double exp(double x)


cdef double pi=np.pi, twopi = 2*pi, twopi2 = 2*pi**2
cdef double[:] _k, _Pk, _k2Pk
cdef int Nk
_k, _Pktmp, blip = np.loadtxt('/home/ccc/Documents/prog/correlations/correlations/data/power.dat', skiprows=1).T
_Pk = _Pktmp * twopi2 * 4*np.pi
Nk = len(_k)
_k2Pk = np.array(_k)**2 * np.array(_Pk)
cdef eightpi3 = 8 * pi**3

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


################################################################################
# Filters
@cython.cdivision(True)
cdef double WG(double x):
    return exp(-x**2/2)

@cython.cdivision(True)
cdef double WTH(double x):
    cdef double x2
    x2 = x**2
    if x < 1e-4:
        return 1 - x2/10
    else:
        return 3. * (sin(x)/x2/x - cos(x)/x2)


################################################################################
# Power spectrum handling
@cython.cdivision(True)
cdef double Pk_power_law(double k):
    return k**-2

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef double Pk_CDM(double k):
    cdef int il = 0, ir = Nk, im
    cdef double km

    if k > _k[Nk-1] or k < _k[0]:
        return 0

    while il + 1 < ir:
        im = (il + ir) // 2
        km = _k[im]

        if k < km:
            ir = im
        else:
            il = im

    cdef Pkl, Pkr, kl, kr
    kl = _k[il]
    kr = _k[ir]
    Pkl = _Pk[il]
    Pkr = _Pk[ir]

    return (k - kl) / (kr - kl) * (Pkr - Pkl) + Pkl

cdef double Pk(double k):
    return Pk_CDM(k)

################################################################################
# Compute sigma
@cython.cdivision(True)
cpdef double _sigma_integrand_TH(double k, int i, double R):
    return k**(2 + i*2) * Pk(k) * WTH(k*R)**2 / twopi2

@cython.cdivision(True)
cpdef double _sigma_integrand_G(double k, int i, double R):
    return k**(2 + i*2) * Pk(k) * WG(k*R)**2 / twopi2

cpdef sigma_TH(int i, double R):
    cdef double sigma2, err
    sigma2, err = quad(_sigma_integrand_TH, _k[0], _k[Nk-1], (i, R), epsrel=1e-4)
    return sqrt(sigma2) 

cpdef sigma_G(int i, double R):
    cdef double sigma2, err
    sigma2, err = quad(_sigma_integrand_G, _k[0], _k[Nk-1], (i, R), epsrel=1e-4)
    return sqrt(sigma2)    


################################################################################
# Compute integrand
cpdef double integrand(
       double k, double phi, double theta, int ikx, int iky, int ikz, int ikk,
       double dx, double dy, double dz, double R1, double R2):
    cdef double sin_theta, cos_theta, ksin, kx, ky, kz
    cdef int ii
    cdef double exppart, intgd, k2Pk

    k2Pk = k**2 * Pk(k)

    sin_theta = sin(theta)
    cos_theta = cos(theta)
    ksin = k * sin_theta

    kx = ksin * cos(phi)
    ky = ksin * sin(phi)
    kz = k * cos_theta

    # Compute parity
    ii = ikx + iky + ikz - ikk

    if ii % 2 == 0:
        exppart = (-1)**(ii//2) * cos((kx * dx + ky * dy + kz * dz))
    else:
        exppart = (-1)**((ii-1)//2) * sin((kx * dx + ky * dy + kz * dz))

    intgd = (
        k2Pk * sin_theta *
        exppart *
        WG(k * R1) * WG(k * R2)
    )
    if ikx != 0:
        intgd *= kx**ikx
    if iky != 0:
        intgd *= ky**iky
    if ikz != 0:
        intgd *= kz**ikz
    if ikk != 0:
        intgd /= k**ikk

    return intgd / eightpi3

cpdef double integrand_lambdaCDM(double phi, double theta, int ikx, int iky, int ikz, int ikk,
                                double dx, double dy, double dz, double R1, double R2):
    '''
    Compute the integral of the correlation along the k direction
    using trapezoidal rule.
    '''
    cdef double sin_theta, cos_theta, ksin, kx, ky, kz
    cdef int ii
    cdef double exppart, intgd, k, kprev, cur, prev

    sin_theta = sin(theta)
    cos_theta = cos(theta)

    intgd = 0
    cur = 0
    prev = 0
    # Compute parity
    ii = ikx + iky + ikz - ikk

    for i in range(Nk):
        k = _k[i]
        kx = k * sin_theta * cos(phi)
        ky = k * sin_theta * sin(phi)
        kz = k * cos_theta

        if ii % 2 == 0:
            exppart = (-1)**(ii//2) * cos((kx * dx + ky * dy + kz * dz))
        else:
            exppart = (-1)**((ii-1)//2) * sin((kx * dx + ky * dy + kz * dz))
        
        cur = (
            _k2Pk[i] * sin_theta * exppart * WG(k * R1) * WG(k * R2)
        )

        if ikx != 0:
            cur *= kx**ikx
        if iky != 0:
            cur *= ky**iky
        if ikz != 0:
            cur *= kz**ikz
        if ikk != 0:
            cur /= k**ikk

        if i > 0:
            intgd += (k - kprev) * (prev + cur) / 2

        kprev = k
        prev = cur

    return intgd / eightpi3

def compute_correlation(int ikx, int iky, int ikz, int ikk,
                        double dx, double dy, double dz, double R1, double R2,
                        int sign1, int sign2, int nsigma1, int nsigma2,
                        str Pkchoice
    ):
    cdef double s1s2
    cdef double kmin = _k[0], kmax = _k[Nk-1]

    s1s2 = sign1 * sigma_G(nsigma1, R1) * sign2 * sigma_G(nsigma2, R2)

    if Pkchoice == 'power-law':
        res, _ = tplquad(integrand,
                         0, pi,                                   # boundaries on theta
                         lambda theta: 0, lambda theta: twopi,              # boundaries on phi
                         lambda theta, phi: kmin, lambda theta, phi: kmax,  # boundaries on k
                         epsrel=1e-6, args=(ikx, iky, ikz, ikk, dx, dy, dz, R1, R2)
        )
    elif Pkchoice == 'Lambda-CDM':
        res, _ = dblquad(integrand_lambdaCDM,
                         0, pi, lambda theta: 0, lambda theta: twopi,
                         epsrel=1e-6, args=(ikx, iky, ikz, ikk, dx, dy, dz, R1, R2)
        )
    return res / s1s2


# @cython.boundscheck(False)
# cpdef double integrand_cubature(double[:] xarray, int ikx, int iky, int ikz, int ikk,
#                                 double dx, double dy, double dz, double R1, double R2):
#     cdef double k, phi, theta
#     k = xarray[0]
#     phi = xarray[1]
#     theta = xarray[2]

#     return integrand(xarray[0], xarray[1], xarray[2], ikx, iky, ikz, ikk,
#                      dx, dy, dz, R1, R2)


# def compute_correlation_cubature(int ikx, int iky, int ikz, int ikk,
#                                  double dx, double dy, double dz, double R1, double R2,
#                                  int sign1, int sign2, int nsigma1, int nsigma2):
#     cdef double s1s2
#     cdef double kmin = _k[0], kmax = _k[Nk-1]
#     s1s2 = sign1 * sigma_G(nsigma1, R1) * sign2 * sigma_G(nsigma2, R2)
#     cdef double[3] xmin = [_k[0], 0, 0], xmax = [_k[-1], pi, twopi]
    
#     res, _ = cubature(
#         integrand_cubature, 3, 1, xmin, xmax, 'pcubature', 1e-7, 1e-4, 0, 0,
#         args=(ikx, iky, ikz, ikk, dx, dy, dz, R1, R2)
#     )
#     return res / s1s2 / (8*np.pi**3)
