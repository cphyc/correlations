import numpy as np
from numpy import sqrt, pi, cos, sin
from scipy.integrate import dblquad
from numba import jit
from itertools import combinations_with_replacement
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import lru_cache, partial
from multiprocessing import Pool, cpu_count
import os

from .utils import integrand_lambdaCDM as integrand_cython

this_dir, this_filename = os.path.split(__file__)

k, Pk, _ = np.loadtxt(os.path.join(this_dir, 'data', 'power.dat'),
                      skiprows=1).T
Pk *= 2*np.pi**2 * 4*np.pi
k = k[::1]
Pk = Pk[::1]

# Pk = k**-2

twopi2 = 2 * np.pi**2
dk = np.diff(k)


def sigma(i, R):
    # Note: below we use exp(-(kR)**2) == exp(-(kR)**2/2) **2
    integrand = k**(2*i+2) * Pk * np.exp(-(k * R)**2) / twopi2
    return sqrt(np.sum((integrand[1:] + integrand[:-1]) * dk) / 2)


def odd(i):
    return (i % 2) == 1


@lru_cache(maxsize=None)
def _correlation(ikx, iky, ikz, ikk, dx, dy, dz, R1, R2,
                 sign1, sign2,
                 nsigma1, nsigma2):
    dX = np.array([dx, dy, dz])
    # Compute sigmas

    s1s2 = (
        sign1 * sigma(nsigma1, R1) *
        sign2 * sigma(nsigma2, R2))

    # Integrate
    res = dblquad(
        integrand_cython,
        0, pi,                      # theta bounds
        lambda theta: 0, lambda theta: 2*pi,  # phi bounds
        epsrel=1e-3, epsabs=1e-6,
        args=(ikx, iky, ikz, ikk, dX, R1, R2))[0]

    # Divide by sigmas
    res /= s1s2
    return res


def _compute_one(i1i2, X,
                 kxfactor, kyfactor, kzfactor, kfactor,
                 signs, sigma_f, RR, constrains):
    i1, i2 = i1i2
    # if np.isinf(constrains[i1]) or np.isinf(constrains[i2]):
    #     return i1, i2, np.nan

    dX = X[i2, :] - X[i1, :]
    d = sqrt(sum(dX**2))
    ikx, iky, ikz = (kxfactor[i1]+kxfactor[i2],
                     kyfactor[i1]+kyfactor[i2],
                     kzfactor[i1]+kzfactor[i2])
    ikk = kfactor[i1]+kfactor[i2]

    # Odd terms at 0 correlations aren't correlated
    if d == 0 and (odd(ikx) or odd(iky) or odd(ikz)):
        return i1, i2, 0

    # Remove odd terms in direction perpendicular to separation
    if ( (np.dot(dX, [1, 0, 0]) == 0 and odd(ikx)) or
         (np.dot(dX, [0, 1, 0]) == 0 and odd(iky)) or
         (np.dot(dX, [0, 0, 1]) == 0 and odd(ikz))):
        return i1, i2, 0

    R1 = RR[i1]
    R2 = RR[i2]

    sign = (-1)**(kxfactor[i2] + kyfactor[i2] + kzfactor[i2])
    res = _correlation(ikx, iky, ikz, ikk,
                       dX[0], dX[1], dX[2],
                       R1, R2,
                       signs[i1], signs[i2],
                       sigma_f[i1], sigma_f[i2]
    )
    return i1, i2, res * sign

# kx = lambda theta, phi: k * sin(theta) * cos(phi)
# ky = lambda theta, phi: k * sin(theta) * sin(phi)
# kz = lambda theta, phi: k * cos(theta)


k2Pk = k**2 * Pk / (8*np.pi**3)
k2 = k**2
dk = np.diff(k)


def python_integrand(phi, theta, ikx, iky, ikz, ikk, dX, R1, R2):
    '''
    Compute the integral of the correlation along the k direction
    using trapezoidal rule.
    '''
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    ksin = k * sin_theta

    dx, dy, dz = dX

    kx = ksin * cos(phi)
    ky = ksin * sin(phi)
    kz = k * cos_theta

    # Compute parity
    ii = ikx + iky + ikz - ikk

    exppart = (np.exp(-1j * (kx * dx + ky * dy + kz * dz))
               * (1j)**(ii)).real
    intgd = (
        k2Pk * sin_theta * (
            kx**ikx *
            ky**iky *
            kz**ikz /
            k**ikk) *
        exppart *
        np.exp(- (k2 * (R1**2 + R2**2) / 2))
    )

    # Trapezoidal rule for integration along k direction
    integral = np.sum((intgd[1:] + intgd[:-1]) * dk) / 2
    return integral


class Correlator(object):
    def __init__(self, nproc=None):
        self.kxfactor = []
        self.kyfactor = []
        self.kzfactor = []
        self.kfactor = []
        self.positions = np.zeros((0, 3))
        self.constrains = []
        self.smoothing_scales = []
        self.labels = []
        self.labels_c = []
        self.signs = []

        self.Npts = 0
        self.k = k
        self.Pk = Pk

        self.nproc = nproc

    def add_point(self, pos, elements, R, constrains={}, name=None):
        '''Add a constrain at position pos with given elements

        Param
        -----
        pos: array_like
            Spatial position of the point
        elements: str list
            See note below for accepted elements
        R: float
            Smoothing scale at the point
        values, dict_like:
            Values of the elements at the point, indexed by the
            element type (e.g. {'density': 1.68}).
        name, optional, str:
            Name to give to the point. By default, number it.

        Note
        ----
        When setting constrains, be careful to respect the number of
        degrees of freedom. For example, there is 1 dof for the
        density, 3 for its gradient and 6 for the hessian.  The
        convention for the hessian is xx, yy, zz, xy, xz, yz (diagonal
        first, then off-diagonal).

        Accepted elements are:
        * potential (phi): the gravitational potential
        * acceleration (a): the acceleration
        * tide: the tidal tensor
        * density (delta): the overdensity
        * density_density (grad_delta): the gradient of the density
        * hessian: the hessian of the density
        '''
        self.Npts += 1
        # K factors
        kx = []
        ky = []
        kz = []
        kk = []
        cons = []
        smoothing_scales = []
        new_pos = []
        labels = []
        sign = []

        def add(e, n):
            if e in constrains:
                cons.extend(constrains[e])
            else:
                cons.extend([np.nan] * n)
            smoothing_scales.extend([R] * n)
            new_pos.extend([pos] * n)

        # Compute k factors
        for e in elements:
            if e in ['phi', 'potential']:  # Potential
                kx += [0]
                ky += [0]
                kz += [0]
                kk += [2]
                sign += [1]
                labels += [r'$\phi^{%(name)s}$']
                add(e, 1)
            elif e in ['a', 'acceleration']:  # Acceleration
                kx += [1, 0, 0]
                ky += [0, 1, 0]
                kz += [0, 0, 1]
                kk += [2, 2, 2]
                sign += [-1, -1, -1]
                labels += [r'$a_x^{%(name)s}$',
                           r'$a_y^{%(name)s}$',
                           r'$a_z^{%(name)s}$']
                add(e, 3)
            elif e in ['tide']:  # Tidal tensor (with trace!)
                kx += [2, 0, 0, 1, 1, 0]
                ky += [0, 2, 0, 1, 0, 1]
                kz += [0, 0, 2, 0, 1, 1]
                kk += [2, 2, 2, 2, 2, 2]
                sign += [1, 1, 1, 1, 1, 1]
                labels += [r'$q_{xx}^{%(name)s}$', r'$q_{yy}^{%(name)s}$', r'$q_{zz}^{%(name)s}$',
                           r'$q_{xy}^{%(name)s}$', r'$q_{xz}^{%(name)s}$', r'$q_{yz}^{%(name)s}$']
                add(e, 6)
            elif e in ['delta', 'density']:  # Over density
                kx += [0]
                ky += [0]
                kz += [0]
                kk += [0]
                sign += [1]
                labels += [r'$\delta^{%(name)s}$']
                add(e, 1)
            elif e in ['grad_delta', 'density_gradient']:  # Gradient of density
                kx += [1, 0, 0]
                ky += [0, 1, 0]
                kz += [0, 0, 1]
                kk += [0, 0, 0]
                sign += [1, 1, 1]
                labels += [r'$\nabla_x \delta^{%(name)s}$',
                           r'$\nabla_y \delta^{%(name)s}$',
                           r'$\nabla_z \delta^{%(name)s}$']
                add(e, 3)
            elif e in ['hessian']:  # Hessian of density
                kx += [2, 0, 0, 1, 1, 0]
                ky += [0, 2, 0, 1, 0, 1]
                kz += [0, 0, 2, 0, 1, 1]
                kk += [0, 0, 0, 0, 0, 0]
                sign += [1, 1, 1, 1, 1, 1]
                labels += [r'$h_{xx}^{%(name)s}$', r'$h_{yy}^{%(name)s}$', r'$h_{zz}^{%(name)s}$',
                           r'$h_{xy}^{%(name)s}$', r'$h_{xz}^{%(name)s}$', r'$h_{yz}^{%(name)s}$']
                add(e, 6)
            else:
                print('Do not know %s.' % e)

        self.positions = np.concatenate((self.positions, np.array(new_pos)))
        self.kxfactor.extend(kx)
        self.kyfactor.extend(ky)
        self.kzfactor.extend(kz)
        self.kfactor.extend(kk)
        self.constrains.extend(cons)
        self.smoothing_scales.extend(smoothing_scales)
        self.signs.extend(sign)

        # Format the labels
        self.labels.extend((l % {'name': (name if name is not None else
                                          str(self.Npts))}
                            for l in labels))

        Nlabel = len(self.labels)

        self.labels_c = [self.labels[i] for i in range(Nlabel)
                         if np.isnan(self.constrains[i])]

        return self.Npts-1

    @property
    def cov(self):
        '''
        The (unconstrained) covariance matrix
        '''
        if hasattr(self, '_covariance'):
            return self._covariance

        self.compute_covariance()
        return self._covariance

    def compute_covariance(self):
        '''
        Computes the unconstrained covariance matrix.
        '''
        # Create arrays of factors of k and degree of sigma
        kxfactor = self.kxfactor
        kyfactor = self.kyfactor
        kzfactor = self.kzfactor
        kfactor = self.kfactor
        sigma_f = [ikx + iky + ikz - ikk
                   for ikx, iky, ikz, ikk
                   in zip(kxfactor, kyfactor, kzfactor, kfactor)]
        RR = self.smoothing_scales
        signs = np.asarray(self.signs)

        # Number of elements in the covariance matrix
        Ndim = len(kxfactor)
        # Create an array with all the positions of shape (Ndim, 3)
        X = self.positions

        cov = np.zeros((Ndim, Ndim))

        # Loop on all combinations of terms, with replacement
        iterator = combinations_with_replacement(range(Ndim), 2)

        fun = partial(_compute_one,
                      X=X,
                      kxfactor=kxfactor, kyfactor=kyfactor, kzfactor=kzfactor,
                      kfactor=kfactor, signs=signs, sigma_f=sigma_f,
                      RR=RR, constrains=self.constrains)

        with Pool(self.nproc) as p:
            if self.nproc:
                print('Running with %s processes.' % self.nproc)
            else:
                print('Running with %s processes.' % cpu_count())

            for i1, i2, value in tqdm(p.imap_unordered(
                    fun, iterator,
                    chunksize=Ndim//2),
                                      total=Ndim*(Ndim+1)//2):
                cov[i1, i2] = cov[i2, i1] = value

        self._covariance = cov

        if any(np.linalg.eigvalsh(cov) <= 0):
            print('WARNING: the covariance matrix is not positive definite.')
        return cov

    @property
    def cov_c(self):
        '''
        The (constrained) covariance matrix
        '''
        self._mean_c, self._cov_c = self.constrain()

        return self._cov_c

    @property
    def mean_c(self):
        # This computes the mean too
        self._mean_c, self._cov_c = self.constrain()
        return self._mean_c

    def constrain(self, mean=None):
        '''
        Compute the covariance matrix subject to the linear constrains
        '''
        if mean is None:
            mean = np.zeros(len(self.kxfactor))

        # Set the gradients to 0 at the constrained points
        cons = np.asarray(self.constrains)

        mean_c, cov_c = constrain(mean, self.cov, cons)
        return mean_c, cov_c

    def _plot_cov(self, cov, labels, *args, symlog=False, **kwa):
        '''
        Covariance plot helper.
        '''

        if symlog:
            tmp = np.abs(self.cov_c)
            # Get minimal non null value
            vmin = np.nanmin(np.where(tmp == 0, np.nan, tmp))

            # Round it to closer power of 10
            vmin = 10**(np.floor(np.log10(vmin)))

            kwa.update({'norm': mpl.colors.SymLogNorm(vmin)})

        plt.imshow(cov, cmap='seismic', vmin=-1, vmax=1, *args, **kwa)
        N, _ = cov.shape

        ticks = np.arange(N)
        plt.xticks(ticks, labels)
        plt.yticks(ticks, labels)

    def plot_cov(self, *args, **kwa):
        '''Plot the covariance matrix.'''
        self._plot_cov(self.cov, self.labels, *args, **kwa)

    def plot_cov_c(self, *args, **kwa):
        '''Plot the constrained matrix.'''
        self._plot_cov(self.cov_c, self.labels_c, *args, **kwa)


    def _fmt_element(self, element, size=10):
        if np.isclose(element, 0):
            return ' '*size
        else:
            if size < 5:
                raise Exception('Size too small')
            fmtstring = '{{:{}.{}f}}'.format(size, size-5)
            return fmtstring.format(element)

    def describe(self):
        cov = self.cov
        print('#'*80)
        print('Covariance matrix')
        for i in range(cov.shape[0]):
            if i == 0:
                print('/  ', end='')
            elif i == cov.shape[0]-1:
                print('\\  ', end='')
            else:
                print('|  ', end='')
            for j in range(cov.shape[1]):
                print(self._fmt_element(cov[i, j]))

            if i == 0:
                print('  \\')
            elif i == cov.shape[0]-1:
                print('  /')
            else:
                print('  |')

    def describe_table(self, order=None, size=10, constrained=False):
        from IPython.display import display, Markdown

        _cov = self.cov_c if constrained else self.cov
        _labels = self.labels_c if constrained else self.labels
        if order is None:
            N = len(_cov)
            cov = _cov
            labels = _labels
        else:
            N = len(_cov)
            cov = _cov[order][:, order]
            labels = np.array(_labels)[order]

        header = '''
        | | {header} |
        |-|{sep}|
        '''.format(
            header=' | '.join(labels),
            sep='-|-'.join(('' for _ in range(N)))
        )
        table = '\n'.join((l.strip() for l in header.split('\n')))

        for i, l in enumerate(labels):
            line = '|{label}|{content}|\n'.format(
                label=l,
                content=' | '.join((
                    '$%s$' % self._fmt_element(cov[i, j], size)
                    for j in range(N)
                    )
                )
            )

            table += line
        # return table
        return display(Markdown(table))

def constrain(mean, cov, values):
    '''Return the constrained mean and covariance given the values.
    values is an array of same length as mean, with np.nan where you don't want
    any constrain and the value elsewhere.

    Parameters
    ---------
    mean : array_like, (N, )
        The (unconstrained) mean
    cov : array_like, (N, N)
        The (unconstrained) covariance matrix
    values : array_like (N, )
        The values of the constrain. See the notes for the format of this array

    Returns
    -------
    mean_c : array_like, (N-n, )
         The constrained mean
    cov_c : array_like, (N-n, N-n)
         The constrained variance

    Note
    ----
    The `values` array can be filled either with scalars, nan of
    inf. The meaning of these is given in the table below.
    Type   | Meaning
    -------|------------------------
    float  | Value at this location
    nan    | No constrain
    inf    | Drop this location
    '''

    # Keep `nan` elements, constrain finite ones
    cons = np.isfinite(values)
    keep = np.isnan(values)

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

# import numpy as np
# from numba import jit
# from itertools import product
# from _correlations import correlation_functions as cf
# from functools import wraps


# @jit(target='cpu', nopython=True)
# def fill_array(A, mask1, mask2, value):
#     '''Fill array A[mask1, mask2] with value'''
#     ii = 0
#     for i in range(len(mask1)):
#         if mask1[i]:
#             jj = 0
#             for j in range(len(mask2)):
#                 if mask2[j]:
#                     A[j, i] = A[i, j] = value[ii, jj]
#                     jj += 1
#             ii += 1


# correlators = {}


# def correlator(field1, field2, dim1, dim2):
#     def wrapper(fun):
#         @wraps(fun)
#         def wrapped(this, dist, mask1, mask2):
#             fun(this, dist, mask1, mask2)

#         correlators[field1, field2] = wrapped
#         return wrapped
#     return wrapper


# class ConstrainedField:
#     def __init__(self, pos, kinds, R, W, k, Pk):
#         if type(kinds) in (tuple, list):
#             self.pos = None
#             self.kinds = []
#             self.R = []
#             for kind in kinds:
#                 self.pos = (pos if self.pos is None
#                             else np.concatenate((self.pos, pos)))
#                 self.kinds.extend([kind for _ in pos])
#                 self.R.extend(list(R))
#             self.pos = list(self.pos)
#         else:
#             self.pos = list(pos)
#             kind = kinds
#             self.kinds = [kind for _ in pos]
#             self.R = list(R)
#         self.constrains = [np.nan for _ in self.pos]
#         self.N = len(pos)
#         self.k = k
#         self.Pk = Pk
#         self.k2Pk = k**2 * Pk
#         self.W = W

#         self.xi = {}
#         self.chi = {}

#     def compute_coeffs(self):
#         print('Computing correlation coefficients')
#         cf.set_k(self.k, self.Pk)
#         pos = np.array(self.pos)
#         R = np.array(self.R)

#         nn, mm = np.array(product([0, 1, 2], [-2, -1, 0, 1, 2]), dtype=int).T
#         coeffs = cf.compute_coeff(R, pos, nn, mm)
#         for i in range(len(nn)):
#             n, m = nn[i], mm[i]
#             if m < 0:
#                 self.chi[n, m] = coeffs[i, :, :]
#             else:
#                 self.xi[n, m] = coeffs[i, :, :]

#     def add_constrain(self, pos, kind, value, R):
#         self.pos.append(pos)
#         self.kinds.append(kind)
#         self.constrains.append(value)
#         self.R.append(R)

#     def compute_cov(self):
#         print('Computing covariance')
#         N = len(self.pos)
#         cov = np.zeros((N, N))
#         kinds = np.array(self.kinds)
#         pos = np.array(self.pos)

#         epsilon = 1e-3

#         # Compute correlations of same kind
#         for k in np.unique(kinds):
#             mask = kinds == k
#             # Get positions
#             X = pos[mask][:, None, :]
#             Y = pos[mask][None, :, :]
#             DX = Y-X
#             dist = np.linalg.norm(DX, axis=-1)
#             dist[dist == 0] = epsilon

#             if k == 'density':
#                 tmp = self.xi00[mask][:, mask]
#                 fill_array(cov, mask, mask, tmp)
#             for idim, dim in enumerate('xyz'):
#                 if k == 'gradient_%s' % dim:
#                     tmp = (
#                         + self.xi11[mask][:, mask]
#                         - DX[..., idim]*DX[..., idim] * self.chi22[mask][:, mask]) / dist**2
#                     # print(k, tmp, tmp.max(), tmp.min())
#                     fill_array(cov, mask, mask, tmp)

#         # Compute cross-correlations
#         kk = np.unique(kinds)
#         for k1, k2 in product(kk, kk):
#             mask1 = kinds == k1
#             mask2 = kinds == k2
#             X = pos[mask1][:, None, :]
#             Y = pos[mask2][None, :, :]
#             DX = Y-X
#             dist = np.linalg.norm(DX, axis=-1)
#             dist[dist == 0] = epsilon

#             # Density correlations
#             if k1 == 'density':
#                 for idim1, dim1 in enumerate('xyz'):
#                     if k2 == 'gradient_%s' % dim1:
#                         tmp = -DX[..., idim1] * self.chi11[mask1][:, mask2] / dist
#                         # print(k1, k2, tmp.max(), tmp.min())
#                         # No correlation at 0 distance
#                         tmp[dist == epsilon] = 0
#                         fill_array(cov, mask1, mask2, tmp)

#                     for idim2, dim2 in enumerate('xyz'):
#                         if k2 == 'hessian_%s%s' % (dim1, dim2):
#                             tmp = (
#                                 - (idim1 == idim2) * self.chi11[mask1][:, mask2]
#                                 + DX[..., idim1] * DX[..., idim2] * self.chi22[mask1][:, mask2]) / dist**2
#                             # print(k1, k2, tmp.max(), tmp.min())
#                             fill_array(cov, mask1, mask2, tmp)

#             # Gradient correlations
#             for idim1, dim1 in enumerate('xyz'):
#                 if k1 == 'gradient_%s' % dim1:
#                     for idim2, dim2 in enumerate('xyz'):
#                         if idim2 < idim1:
#                             pass
#                         if k2 == 'gradient_%s' % dim2:
#                             tmp = (
#                                 + (idim1 == idim2) * self.chi11[mask1][:, mask2]
#                                 - DX[..., idim1] * DX[..., idim2] * self.chi22[mask1][:, mask2]) / dist**2
#                             # print(k1, k2, tmp.max(), tmp.min())
#                             # The 0-lag correlation at 0 distance is null
#                             if idim1 != idim2:
#                                 tmp[dist == epsilon] = 0
#                             fill_array(cov, mask1, mask2, tmp)
#         # Symmetrize matrix
#         for i in range(N):
#             for j in range(i):
#                 pass # cov[i, j] = cov[j, i]
#         self.cov = cov

#         return cov

#     @correlator('delta', 'delta', 1, 1)
#     def delta_delta(self, DX, dist, mask1, mask2):
#         return self.xi00[mask1, mask2]

#     @correlator('delta', 'grad-delta', 1, 3)
#     def delta_graddelta(self, DX, dist, mask1, mask2):
#         z = np.zeros_like(self.chi11)
#         for idim in range(3):
#             tmp = (
#                 + self.xi11[mask1][:, mask2]
#                 - DX[..., idim]*DX[..., idim] *
#                 self.chi22[mask1][:, mask2]) / dist**2

#         return -np.array([self.chi11[mask1, mask2], z, z]) / d

#     @correlator('delta', 'hessian', 1, 9)
#     def delta_hessian(d):
#         z = np.zeros_like(self.chi11)


#     def compute_constrained_cov(self):
#         if not hasattr(self, 'cov'):
#             self.compute_cov()

#         print('Computing constrained covariance')
#         constrains = np.array(self.constrains)
#         mu = np.zeros_like(self.R)
#         sigma = self.cov
#         mu_c, sigma_c = constrain(mu, sigma, constrains)
#         self.mean_c = mu_c
#         self.cov_c = sigma_c
