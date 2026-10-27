# correlations

Compute covariance matrices and conditional Gaussian statistics for Gaussian
random fields using a bundled cosmological power spectrum. Fields include density,
potential, acceleration, tidal tensors, and density derivatives.

## Installation

Python 3.11 or later is required.
From the repository root, install with [uv](https://docs.astral.sh/uv/):

```sh
uv venv
uv pip install -e '.[dev,plot]'
```

The package uses Hatchling to build a pure-Python wheel. Numerical kernels
compile at runtime with Numba and cache their compiled code across processes;
building the project requires no C compiler, Cython, CMake, or setuptools.
Plotting dependencies are available
through the `plot` extra; use `notebook` for IPython and JupyterLab support.
For a regular installation, use `uv pip install .`.
Python changes are available immediately in editable installations. Numba
recompiles changed kernels on their next invocation.

## Usage

```python
from correlations import Correlator

c = Correlator()
c.add_point([0, 0, 0], ["delta"], R=1, name="A")
c.add_point([3, 0, 0], ["delta"], R=1, name="B")
print(c.cov)  # normalized covariance, with unit density variance

# Condition the density at A to one standard deviation.
c.constrains[c["A"]["delta"]] = 1
print(c.c.mean)
print(c.c.cov)
```

`add_point` accepts `potential` (`phi`), `acceleration` (`a`), `tide`,
`density` (`delta`), `density_gradient` (`grad_delta`), `hessian`, and `third`.
It also accepts the generic potential and normalized density derivatives:

| Order | Potential | Density | Components |
| --- | --- | --- | --- |
| 0 | `phi` | `nu` | 1 |
| 1 | `phi_i` | `nu_i` | 3 |
| 2 | `phi_ij` | `nu_ij` | 6 |
| 3 | `phi_ijk` | `nu_ijk` | 10 |
| 4 | `phi_ijkl` | `nu_ijkl` | 15 |
| 5 | `phi_ijklm` | `nu_ijklm` | 21 |
| n | Continue the suffix `ijklmnopqrstuvwxyz` | Same | `(n+1)(n+2)/2` |

The indices denote the full symmetric Cartesian tensor. Rank two uses
`xx, yy, zz, xy, xz, yz`; rank three uses
`xxx, yyy, zzz, xxy, xxz, yyx, yyz, zzx, zzy, xyz`.
Higher ranks group pure powers first, then mixed powers by descending multiplicity.
`derivative_components(n)` returns the exact order as `(nx, ny, nz)` exponent
triples, useful for constructing constraints on high-order tensors.

Density derivatives of order n divide by **σₙ(R)**, consistent with the original
fields; only `nu` itself is δ/σ₀. Potential derivatives divide by **σₙ₋₂(R)**.
These are spectral-moment normalizations: individual vector components, for
example, have variance 1/3. The original potential sign convention is preserved:
`phi_i` is **minus** `acceleration`, and `phi_ij` equals `tide`.
All density aliases have the same signs as their generic names.

```python
from correlations import Correlator, derivative_components

c = Correlator(nproc=4)
c.add_point([0, 0, 0], ["phi_ijklm", "nu_ij"], R=1)
print(derivative_components(5))  # order of the 21 potential components
print(c.cov.shape)  # (27, 27)
```

Constraint values use the existing `constrains` spelling:
finite values impose a constraint, `nan` leaves a component free, and infinity
drops it from the conditional result. The standalone `constrain(mean, cov, values)`
function also accepts ordinary lists.

## Covariance backends

The default `backend="numba"` performs exact angular integration using a finite
spherical-Bessel expansion, followed by Numba-compiled CPU accumulation on the
original radial grid. Tensor components and rotated separations share scalar
radial moments depending only on distance, smoothing, and inverse-k power.
For potential derivatives through order three, seven moments assemble each
20 × 20 point-pair block. Compiled planning and matrix assembly reuse those
moments across the entire matrix, using exact float keys without rounding
positions. It retains the reference trapezoidal rule and spectral moments, with no
angular quadrature settings to tune. `nproc=4` uses four Numba threads; the default
uses one. `quiet=False` displays progress over geometry batches.

The first call compiles or loads Numba kernels; subsequent calls avoid that setup.
Numba caches compiled code, while `.cov` caches the resulting matrix until a point
is added. A forced `compute_covariance()` call recomputes it.

The original implementation remains available for numerical reference:

```python
reference = c.compute_covariance(backend="reference")
accelerated = c.compute_covariance(backend="numba")
# Or select a persistent default with Correlator(backend="reference").
```

`compute_covariance_py()` also retains the original behavior. This reference
backend uses Numba integrands, adaptive SciPy angular integration, joblib caching
in `./cachedir`, and multiprocessing when `nproc>1`. Only load pickle caches from
trusted sources. The new backend does not require a GPU or GPU libraries.
Like the reference, accuracy at large separation and high derivative order is
ultimately limited by the bundled radial grid; exact angular integration does not
refine that grid. Dense covariance storage still scales quadratically in the
number of tensor components.

## Development

```sh
uv run --no-sync python -m pytest -m 'not slow'
uv run --no-sync python -m pytest -m slow
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync python -m build
uv run --no-sync pre-commit install
uv run --no-sync python benchmarks/benchmark_covariance.py --threads 4
uv run --no-sync python benchmarks/benchmark_line.py
```

The slow suite compares independent multidimensional quadrature methods and can
take considerably longer than the core tests. Warnings are treated as errors.
The `--no-sync` commands use the environment installed above with `uv pip`.
The benchmark excludes Numba startup from its repeated timings and bypasses the
reference's joblib result cache. It reports both CPU timings and numerical error.
Add `--profile /tmp/covariance.prof` to profile the reference call; profiling
overhead then contributes to the reported reference time.

