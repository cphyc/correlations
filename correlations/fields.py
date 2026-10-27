"""Fourier factors for symmetric Cartesian derivatives.

The generic suffixes i, ij, ijk, ... denote complete symmetric tensors,
not particular Cartesian components. Components are grouped by multiplicity:
pure powers first, then permutations of mixed powers in x, y, z order.
"""

from functools import lru_cache
from itertools import permutations


def _partition(n, maximum=None):
    if n == 0:
        yield ()
        return
    for first in range(min(n, n if maximum is None else maximum), 0, -1):
        for rest in _partition(n - first, first):
            if len(rest) < 3:
                yield (first, *rest)


@lru_cache(None)
def derivative_components(order):
    """Return exponent triples, preserving the existing rank 0–3 ordering."""
    if order < 0:
        raise ValueError("Derivative order must be nonnegative")
    if order == 0:
        return ((0, 0, 0),)
    components = []
    for partition in _partition(order):
        for axes in permutations(range(3), len(partition)):
            triple = [0, 0, 0]
            for axis, power in zip(axes, partition, strict=True):
                triple[axis] = power
            if tuple(triple) not in components:
                components.append(tuple(triple))
    return tuple(components)


_ALIASES = {
    "potential": ("phi", 0, 1),
    "phi": ("phi", 0, 1),
    "a": ("phi", 1, -1),
    "acceleration": ("phi", 1, -1),
    "tide": ("phi", 2, 1),
    "delta": ("nu", 0, 1),
    "density": ("nu", 0, 1),
    "grad_delta": ("nu", 1, 1),
    "density_gradient": ("nu", 1, 1),
    "hessian": ("nu", 2, 1),
    "third": ("nu", 3, 1),
}


@lru_cache(None)
def field_spec(name):
    """Return (components, inverse-k power, sign) for a field name.

    Potential follows the original project's Fourier convention: phi has
    factor (ik)^(-2), so phi_i = -acceleration and phi_ij = tide.
    All fields divide by sigma(order - inverse_k_power, R).
    """
    if name in _ALIASES:
        family, order, sign = _ALIASES[name]
    else:
        family, separator, suffix = name.partition("_")
        if family not in ("phi", "nu") or (
            separator and (not suffix or suffix != "ijklmnopqrstuvwxyz"[: len(suffix)])
        ):
            raise ValueError(f"Unsupported field name: {name!r}")
        order = len(suffix)
        sign = 1
    return derivative_components(order), 2 if family == "phi" else 0, sign
