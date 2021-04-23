from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

try:
    import numpy as np
    import cython
    include_dirs = [np.get_include(), 'correlations']
except ImportError:
    raise ImportError(
"""Could not import cython or numpy. Building this package from source requires
cython and numpy to be installed. Please install these packages using
the appropriate package manager for your python environment.""")

cython_extensions = [
    Extension("correlations.utils",
              ["correlations/utils.pyx"],
              include_dirs=include_dirs)
]

setup(
    ext_modules=cythonize(cython_extensions)
)
