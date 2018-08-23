import numpy as np
from numpy import sqrt, pi, cos, sin
from scipy.integrate import dblquad
from itertools import combinations_with_replacement
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import wraps, partial
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import os

from .utils import Utils
from .fortran_utils import compute_covariance
from .funcs import LRUCache


this_dir, this_filename = os.path.split(__file__)

k, Pk, _ = np.loadtxt(os.path.join(this_dir, 'data', 'power.dat'),
                      skiprows=1).T
Pk *= 2*np.pi**2 * 4*np.pi
k = k[::1]
Pk = Pk[::1]

utils = Utils(k, Pk)
sigma = utils.sigma
integrand_cython = utils.integrand_lambdaCDM

# Pk = k**-2

twopi2 = 2 * np.pi**2
dk = np.diff(k)


def odd(i):
    return (i % 2) == 1


@LRUCache()
def _correlation(ikx, iky, ikz, ikk, dx, dy, dz, R1, R2,
                 sign1, sign2,
                 nsigma1, nsigma2):
    # Compute sigmas

    s1s2 = (
        sign1 * sigma(nsigma1, R1) *
        sign2 * sigma(nsigma2, R2))

    # Integrate
    res = dblquad(
        integrand_cython,
        0, pi,                      # theta bounds
        lambda theta: 0, lambda theta: 2*pi,  # phi bounds
        epsrel=1e-3, epsabs=1e-5,
        args=(ikx, iky, ikz, ikk, dx, dy, dz, R1, R2))[0]

    # # Integrate
    # res2 = nquad(
    #     integrand_ython,
    #     ((0, 2*pi), (0, pi)),
    #     args=(ikx, iky, ikz, ikk, *dX, R1, R2))[0]

    # assert np.isclose(res, res2)

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


def integrand_python(phi, theta, ikx, iky, ikz, ikk, dx, dy, dz, R1, R2):
    '''
    Compute the integral of the correlation along the k direction
    using trapezoidal rule.
    '''
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    ksin = k * sin_theta

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


class CovarianceAccessor(object):
    def __init__(self, correlator):
        self.parent = correlator

    @property
    def cov(self):
        return self.parent.cov_c

    @property
    def mean(self):
        return self.parent.mean_c

    def __getitem__(self, key):
        mapping = self.parent._mapping
        ret = np.array(
            [mapping[v][key] for v in mapping
             if key in mapping[v]])

        # Now recompute indexes
        constrains = self.parent.constrains
        old_to_new = np.zeros_like(constrains) * np.nan

        j = 0
        for i, c in enumerate(constrains):
            if np.isnan(c):
                old_to_new[i] = j
                j += 1

        ret = old_to_new[ret]
        return ret


class Correlator(object):
    '''A class to compute correlation matrix at arbitrary points using
    a real power spectrum.

    Params
    ------
    nproc : int, optional
       Use this number of processor. If None, use only one.
    quiet : boolean, optional
       Turn off notification

    To add points, use the `add_point` method.
    To access the mean and covariance use `.mean` and `.cov`
    To access the constrained mean and covariance, use `.c.mean` and `.c.cov`.

    You can also access the offsets to tweak e.g. the constrains using
    array syntax. For example
        >> c = Correlator()
        .. c.add_point([ 0, 0, 0], ['delta'], 1, name="A")
        .. c.add_point([10, 0, 0], ['delta'], 1, name="B")

        >> c["A"]
        {'delta': array([0])}

        >> c["delta"]
        array([[0],
               [1]])

    You can also access the offsets within the correlated data using
    the same format e.g. `c.c["A"]`.
    '''
    def __init__(self, nproc=None, quiet=False):
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
        self._mapping = defaultdict(dict)
        self.c = CovarianceAccessor(self)

        self.Npts = 0
        self.k = k
        self.Pk = Pk

        self.nproc = nproc if nproc else 1

        self._covariance_valid = False
        self.quiet = quiet
        if quiet:
            def pbar(iterable, *args, **kwa):
                return iterable
            self._pbar = pbar
        else:
            self._pbar = tqdm

    def invalidate_covariance(fun):
        @wraps(fun)
        def wrapper(self, *args, **kwa):
            self._covariance_valid = False
            return fun(self, *args, **kwa)

        return wrapper

    @invalidate_covariance
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
        name = name if name else str(self.Npts)

        i0 = len(self.constrains)

        pos = np.asarray(pos)
        if pos.ndim > 1:
            raise Exception('pos argument should have ndim=1')

        # Helper function: add the contrain, smoothing scale and
        # position to the relevant arrays
        def add(e, n):
            istart = len(cons) + i0
            if e in constrains:
                cons.extend(constrains[e])
            else:
                cons.extend([np.nan] * n)
            iend = len(cons) + i0
            self._mapping[name][e] = np.arange(istart, iend)
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
        self.labels.extend((l % {'name': name}
                            for l in labels))

        Nlabel = len(self.labels)

        self.labels_c = [self.labels[i] for i in range(Nlabel)
                         if np.isnan(self.constrains[i])]

        return self.Npts-1

    def get_offset_by_name(self, name):
        return self._mapping[name]

    def get_offset_by_value(self, key):
        ret = np.array(
            [self._mapping[v][key] for v in self._mapping
             if key in self._mapping[v]])

        return ret

    def __getitem__(self, key):
        if type(key) == str:
            if key in self._mapping:
                return self.get_offset_by_name(key)
            else:
                return self.get_offset_by_value(key)
        else:
            raise Exception('Did not understand.')

    @property
    def points(self):
        return list(self._mapping.keys())

    @property
    def cov(self):
        '''
        The (unconstrained) covariance matrix
        '''
        if self._covariance_valid:
            return self._covariance

        self.compute_covariance_py()
        return self._covariance

    def compute_covariance(self):
        return self.compute_covariance_py()

    def compute_covariance_f90(self):
        # Create arrays of factors of k and degree of sigma
        kxfactor = self.kxfactor
        kyfactor = self.kyfactor
        kzfactor = self.kzfactor
        kfactor = self.kfactor
        RR = self.smoothing_scales
        signs = np.asarray(self.signs)

        # Create an array with all the positions of shape (Ndim, 3)
        X = self.positions

        cov = compute_covariance(
            self.k, self.Pk, X, RR, kxfactor, kyfactor, kzfactor, kfactor,
            signs)
        self._covariance = cov
        self._covariance_valid = True
        return cov

    def compute_covariance_py(self):
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

        if self.nproc > 1:
            with Pool(self.nproc) as p:
                if self.nproc and not self.quiet:
                    print('Running with %s processes.' % self.nproc)
                elif not self.quiet:
                    print('Running with %s processes.' % cpu_count())

                for i1, i2, value in self._pbar(
                        p.imap_unordered(
                            fun, iterator,
                            chunksize=Ndim//2),
                        total=Ndim*(Ndim+1)//2):

                    cov[i1, i2] = cov[i2, i1] = value
        else:
            for i1, i2 in self._pbar(iterator, total=Ndim*(Ndim+1)//2):
                _, _, value = fun((i1, i2))
                cov[i1, i2] = cov[i2, i1] = value
        self._covariance = cov

        if any(np.linalg.eigvalsh(cov) <= 0):
            print('WARNING: the covariance matrix is not positive definite.')

        self._covariance_valid = True
        return cov

    @property
    def cov_c(self):
        '''
        The (constrained) covariance matrix.
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
