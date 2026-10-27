from collections import defaultdict
from functools import partial, wraps
from itertools import combinations_with_replacement
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from joblib import Memory
from numpy import cos, pi, sin, sqrt
from scipy.integrate import dblquad
from tqdm import tqdm

from .fields import field_spec
from .utils import Utils

location = "./cachedir"
memory = Memory(location, verbose=0)

k, Pk, _ = np.loadtxt(Path(__file__).parent / "data" / "power.dat", skiprows=1).T
Pk *= 2 * np.pi**2 * 4 * np.pi

utils = Utils(k, Pk)
sigma = utils.sigma
integrand_numba = utils.integrand_lambdaCDM

twopi2 = 2 * np.pi**2
dk = np.diff(k)


def odd(i):
    return (i % 2) == 1


@memory.cache
def _correlation(
    ikx, iky, ikz, ikk, dx, dy, dz, R1, R2, sign1, sign2, nsigma1, nsigma2
):
    # Compute sigmas
    s1s2 = sign1 * sigma(nsigma1, R1) * sign2 * sigma(nsigma2, R2)

    # Integrate
    res = dblquad(
        integrand_numba,
        0,
        pi,  # theta bounds
        lambda theta: 0,
        lambda theta: 2 * pi,  # phi bounds
        epsrel=1e-3,
        epsabs=1e-5,
        args=(ikx, iky, ikz, ikk, dx, dy, dz, R1, R2),
    )[0]

    # Divide by sigmas
    res /= s1s2
    return res


def _compute_one(
    i1i2, X, kxfactor, kyfactor, kzfactor, kfactor, signs, sigma_f, RR, constrains
):
    i1, i2 = i1i2
    dX = X[i2, :] - X[i1, :]
    d = sqrt(sum(dX**2))
    ikx, iky, ikz = (
        kxfactor[i1] + kxfactor[i2],
        kyfactor[i1] + kyfactor[i2],
        kzfactor[i1] + kzfactor[i2],
    )
    ikk = kfactor[i1] + kfactor[i2]

    # Odd terms at 0 correlations aren't correlated
    if d == 0 and (odd(ikx) or odd(iky) or odd(ikz)):
        return i1, i2, 0

    # Remove odd terms in direction perpendicular to separation
    if (
        (np.dot(dX, [1, 0, 0]) == 0 and odd(ikx))
        or (np.dot(dX, [0, 1, 0]) == 0 and odd(iky))
        or (np.dot(dX, [0, 0, 1]) == 0 and odd(ikz))
    ):
        return i1, i2, 0

    R1 = RR[i1]
    R2 = RR[i2]

    sign = (-1) ** (kxfactor[i2] + kyfactor[i2] + kzfactor[i2])
    res = _correlation(
        ikx,
        iky,
        ikz,
        ikk,
        dX[0],
        dX[1],
        dX[2],
        R1,
        R2,
        signs[i1],
        signs[i2],
        sigma_f[i1],
        sigma_f[i2],
    )
    return i1, i2, res * sign


k2Pk = k**2 * Pk / (8 * np.pi**3)
k2 = k**2
dk = np.diff(k)


def integrand_python(phi, theta, ikx, iky, ikz, ikk, dx, dy, dz, R1, R2):
    """
    Compute the integral of the correlation along the k direction
    using trapezoidal rule.
    """
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    ksin = k * sin_theta

    kx = ksin * cos(phi)
    ky = ksin * sin(phi)
    kz = k * cos_theta

    # Compute parity
    ii = ikx + iky + ikz - ikk

    exppart = (np.exp(-1j * (kx * dx + ky * dy + kz * dz)) * (1j) ** (ii)).real
    intgd = k2Pk * sin_theta * (kx**ikx * ky**iky * kz**ikz / k**ikk) * exppart

    if R1 + R2 > 0:
        intgd *= np.exp(-(k2 * (R1**2 + R2**2) / 2))

    # Trapezoidal rule for integration along k direction
    integral = np.sum((intgd[1:] + intgd[:-1]) * dk) / 2
    return integral


class CovarianceAccessor:
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
        ret = np.array([mapping[v][key] for v in mapping if key in mapping[v]])

        # Now recompute indexes
        constrains = self.parent.constrains
        old_to_new = np.zeros_like(constrains) * np.nan

        j = 0
        for i, c in enumerate(constrains):
            if np.isnan(c):
                old_to_new[i] = j
                j += 1

        ret = old_to_new[ret]

        # Filter out bad lines
        mask = ~np.all(np.isnan(ret), axis=1)
        ret = ret[mask]

        # Cast to integer
        return np.where(np.isnan(ret), -1, ret).astype(np.int64)


class Correlator:
    """A class to compute correlation matrix at arbitrary points using
    a real power spectrum.

    Params
    ------
    nproc : int, optional
       Use this number of processor. If None, use only one.
    quiet : boolean, optional
       Turn off notification
    backend : {"numba", "reference"}, optional
       Numba CPU accumulation with exact angular integration (default),
       or the original quadrature algorithm with Numba-compiled integrands.

    To add points, use the `add_point` method.
    To access the mean and covariance use `.mean` and `.cov`
    To access the constrained mean and covariance, use `.c.mean` and `.c.cov`.

    You can also access the offsets to tweak e.g. the constrains using
    array syntax.

    Examples
    --------
    >>> c = Correlator()
    ... c.add_point([ 0, 0, 0], ['delta'], 1, name="A")
    ... c.add_point([10, 0, 0], ['delta'], 1, name="B")

    >>> c["A"]
    {'delta': array([0])}

    >>> c["delta"]
    array([[0],
           [1]])

    You can also access the offsets within the correlated data using
    the same format e.g. `c.c["A"]`. Note that invalid positions will
    be written as -1.
    """

    def __init__(self, nproc=None, quiet=True, *, backend="numba"):
        if backend not in ("numba", "reference"):
            raise ValueError("backend must be 'numba' or 'reference'")
        self.backend = backend
        self.kxfactor = []
        self.kyfactor = []
        self.kzfactor = []
        self.kfactor = []
        self.positions = np.zeros((0, 3))
        self.constrains = np.array([])
        self.smoothing_scales = np.array([])
        self.labels = []
        self.labels_c = []
        self.signs = []
        self._mapping = defaultdict(dict)
        self.c = CovarianceAccessor(self)

        self.Npts = 0
        self.k = k
        self.Pk = Pk

        if nproc is not None and (not isinstance(nproc, int) or nproc < 1):
            raise ValueError("nproc must be a positive integer or None")
        self.nproc = nproc if nproc is not None else 1

        self._covariance_valid = False
        self.quiet = quiet
        if quiet:

            def pbar(iterable, *args, **kwa):
                return iterable

            self._pbar = pbar
        else:
            self._pbar = tqdm

    @staticmethod
    def invalidate_covariance(fun):
        @wraps(fun)
        def wrapper(self, *args, **kwa):
            self._covariance_valid = False
            return fun(self, *args, **kwa)

        return wrapper

    @invalidate_covariance
    def add_point(self, pos, elements, R, constrains=None, name=None):
        """Add a constrain at position pos with given elements

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
        * density_gradient (grad_delta): the gradient of the density
        * hessian: the hessian of the density
        * third: the third derivative of the density

        Generic symmetric tensors are phi, phi_i, phi_ij, phi_ijk, ...
        and nu, nu_i, nu_ij, nu_ijk, ... . For order n, phi derivatives
        divide by sigma(n-2, R); density derivatives divide by sigma(n, R).
        The suffix contains abstract indices, not Cartesian coordinates.
        Use derivative_components(n) to obtain component exponent triples.
        """
        constrains = {} if constrains is None else constrains
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            raise ValueError("pos must contain three finite coordinates")
        if not np.isfinite(R) or R < 0:
            raise ValueError("R must be finite and nonnegative")
        elements = [elements] if isinstance(elements, str) else list(elements)
        if not elements:
            raise ValueError("elements must contain supported field names")
        specs = {e: field_spec(e) for e in elements}
        sizes = {e: len(specs[e][0]) for e in elements}
        if len(set(elements)) != len(elements):
            raise ValueError("elements must not contain duplicates")
        constrains = {e: np.atleast_1d(v).astype(float) for e, v in constrains.items()}
        for e, values in constrains.items():
            if e not in elements or values.shape != (sizes[e],):
                raise ValueError(
                    "Each constraint must match its element's number of components"
                )
        if name is not None and name in self._mapping:
            raise ValueError(f"Point name already exists: {name}")
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

        # A rank-n symmetric tensor contains (n+1)(n+2)/2 components.
        for e in elements:
            components, inverse_k_power, field_sign = specs[e]
            symbol = {
                "potential": r"\phi",
                "phi": r"\phi",
                "a": "a",
                "acceleration": "a",
                "tide": "q",
                "delta": r"\delta",
                "density": r"\delta",
                "grad_delta": r"\delta",
                "density_gradient": r"\delta",
                "hessian": "h",
                "third": r"\delta",
            }.get(e, r"\phi" if e.startswith("phi_") else r"\nu")
            for nx, ny, nz in components:
                kx.append(nx)
                ky.append(ny)
                kz.append(nz)
                kk.append(inverse_k_power)
                sign.append(field_sign)
                axes = "x" * nx + "y" * ny + "z" * nz
                if e in ("grad_delta", "density_gradient"):
                    label = r"\nabla_" + axes + r" \delta"
                else:
                    label = symbol + ("_{" + axes + "}" if axes else "")
                labels.append("$" + label + r"^{%(name)s}$")
            add(e, len(components))

        self.positions = np.concatenate((self.positions, np.array(new_pos)))
        self.kxfactor.extend(kx)
        self.kyfactor.extend(ky)
        self.kzfactor.extend(kz)
        self.kfactor.extend(kk)
        self.constrains = np.append(self.constrains, cons)
        self.smoothing_scales = np.concatenate(
            (self.smoothing_scales, smoothing_scales)
        )
        self.signs.extend(sign)

        # Format the labels
        self.labels.extend(label % {"name": name} for label in labels)

        Nlabel = len(self.labels)

        self.labels_c = [
            self.labels[i] for i in range(Nlabel) if np.isnan(self.constrains[i])
        ]

        return self.Npts - 1

    def get_offset_by_name(self, name):
        return self._mapping[name]

    def get_offset_by_value(self, key):
        ret = np.array(
            [self._mapping[v][key] for v in self._mapping if key in self._mapping[v]]
        )

        return ret

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in self._mapping:
                return self.get_offset_by_name(key)
            else:
                return self.get_offset_by_value(key)
        else:
            raise Exception("Did not understand.")

    @property
    def points(self):
        return list(self._mapping.keys())

    @property
    def cov(self):
        """
        The (unconstrained) covariance matrix
        """
        if self._covariance_valid:
            return self._covariance

        self.compute_covariance()
        return self._covariance

    def compute_covariance(self, *, backend=None):
        """Compute covariance with the Numba CPU or original reference backend."""
        backend = self.backend if backend is None else backend
        if backend == "reference":
            return self.compute_covariance_py()
        if backend != "numba":
            raise ValueError("backend must be 'numba' or 'reference'")
        from .fast_covariance import compute_covariance

        cov = compute_covariance(self, nthreads=self.nproc)
        self._covariance = cov
        self._covariance_valid = True
        return cov

    def compute_covariance_py(self):
        """
        Computes the unconstrained covariance matrix.
        """
        # Create arrays of factors of k and degree of sigma
        kxfactor = self.kxfactor
        kyfactor = self.kyfactor
        kzfactor = self.kzfactor
        kfactor = self.kfactor
        sigma_f = [
            ikx + iky + ikz - ikk
            for ikx, iky, ikz, ikk in zip(
                kxfactor, kyfactor, kzfactor, kfactor, strict=True
            )
        ]
        RR = self.smoothing_scales
        signs = np.asarray(self.signs)

        # Number of elements in the covariance matrix
        Ndim = len(kxfactor)
        # Create an array with all the positions of shape (Ndim, 3)
        X = self.positions

        cov = np.zeros((Ndim, Ndim))

        # Loop on all combinations of terms, with replacement
        iterator = combinations_with_replacement(range(Ndim), 2)

        fun = partial(
            _compute_one,
            X=X,
            kxfactor=kxfactor,
            kyfactor=kyfactor,
            kzfactor=kzfactor,
            kfactor=kfactor,
            signs=signs,
            sigma_f=sigma_f,
            RR=RR,
            constrains=self.constrains,
        )

        if self.nproc > 1:
            with Pool(self.nproc) as p:
                if self.nproc and not self.quiet:
                    print(f"Running with {self.nproc} processes.")
                elif not self.quiet:
                    print(f"Running with {cpu_count()} processes.")

                for i1, i2, value in self._pbar(
                    p.imap_unordered(fun, iterator, chunksize=max(1, Ndim // 2)),
                    total=Ndim * (Ndim + 1) // 2,
                ):
                    cov[i1, i2] = cov[i2, i1] = value
        else:
            for i1, i2 in self._pbar(iterator, total=Ndim * (Ndim + 1) // 2):
                _, _, value = fun((i1, i2))
                cov[i1, i2] = cov[i2, i1] = value
        self._covariance = cov

        if any(np.linalg.eigvalsh(cov) <= 0):
            print("WARNING: the covariance matrix is not positive definite.")

        self._covariance_valid = True
        return cov

    @property
    def cov_c(self):
        """
        The (constrained) covariance matrix.
        """
        self._mean_c, self._cov_c = self.constrain()

        return self._cov_c

    @property
    def mean_c(self):
        # This computes the mean too
        self._mean_c, self._cov_c = self.constrain()
        return self._mean_c

    def constrain(self, mean=None):
        """
        Compute the covariance matrix subject to the linear constrains
        """
        if mean is None:
            mean = np.zeros(len(self.kxfactor))

        # Set the gradients to 0 at the constrained points
        cons = np.asarray(self.constrains)

        mean_c, cov_c = constrain(mean, self.cov, cons)
        return mean_c, cov_c

    def _plot_cov(self, cov, labels, *args, symlog=False, **kwa):
        """
        Covariance plot helper.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        if symlog:
            tmp = np.abs(cov)
            # Get minimal non null value
            nonzero = tmp[np.isfinite(tmp) & (tmp > 0)]
            vmin = nonzero.min() if nonzero.size else 1

            # Round it to closer power of 10
            vmin = 10 ** (np.floor(np.log10(vmin)))

            kwa.update({"norm": mpl.colors.SymLogNorm(vmin, vmin=-1, vmax=1)})

        if "norm" not in kwa:
            kwa.setdefault("vmin", -1)
            kwa.setdefault("vmax", 1)
        plt.imshow(cov, *args, cmap="seismic", **kwa)
        N, _ = cov.shape

        ticks = np.arange(N)
        plt.xticks(ticks, labels)
        plt.yticks(ticks, labels)

    def plot_cov(self, *args, **kwa):
        """Plot the covariance matrix."""
        self._plot_cov(self.cov, self.labels, *args, **kwa)

    def plot_cov_c(self, *args, **kwa):
        """Plot the constrained matrix."""
        self._plot_cov(self.cov_c, self.labels_c, *args, **kwa)

    def _fmt_element(self, element, size=10):
        if np.isclose(element, 0):
            return " " * size
        else:
            if size < 5:
                raise Exception("Size too small")
            fmtstring = f"{{:{size}.{size - 5}f}}"
            return fmtstring.format(element)

    def describe(self):
        cov = self.cov
        print("#" * 80)
        print("Covariance matrix")
        for i in range(cov.shape[0]):
            if i == 0:
                print("/  ", end="")
            elif i == cov.shape[0] - 1:
                print("\\  ", end="")
            else:
                print("|  ", end="")
            for j in range(cov.shape[1]):
                print(self._fmt_element(cov[i, j]))

            if i == 0:
                print("  \\")
            elif i == cov.shape[0] - 1:
                print("  /")
            else:
                print("  |")

    def describe_table(self, order=None, size=10, constrained=False):
        from IPython.display import Markdown, display

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

        header = """
        | | {header} |
        |-|{sep}|
        """.format(header=" | ".join(labels), sep="-|-".join("" for _ in range(N)))
        table = "\n".join(line.strip() for line in header.split("\n"))

        for i, label in enumerate(labels):
            line = "|{label}|{content}|\n".format(
                label=label,
                content=" | ".join(
                    f"${self._fmt_element(cov[i, j], size)}$" for j in range(N)
                ),
            )

            table += line
        # return table
        return display(Markdown(table))


def constrain(mean, cov, values):
    """Return the constrained mean and covariance given the values.
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
    """

    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    values = np.asarray(values, dtype=float)
    if (
        mean.ndim != 1
        or values.shape != mean.shape
        or cov.shape != (mean.size, mean.size)
    ):
        raise ValueError("mean and values must have shape (N,) and cov shape (N, N)")

    # Keep `nan` elements, constrain finite ones
    cons = np.isfinite(values)
    keep = np.isnan(values)

    # Keep only finite values (other values: no constrain)
    vals = values[cons]

    Sigma11 = cov[keep][:, keep]
    Sigma12 = cov[keep][:, cons]
    Sigma21 = cov[cons][:, keep]
    Sigma22 = cov[cons][:, cons]

    mu1 = mean[keep]
    mu2 = mean[cons]
    mean_cons = mu1 + Sigma12 @ np.linalg.solve(Sigma22, vals - mu2)
    cov_cons = Sigma11 - Sigma12 @ np.linalg.solve(Sigma22, Sigma21)

    return np.array(mean_cons).flatten(), np.array(cov_cons)
