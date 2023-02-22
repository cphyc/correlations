from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

import cython  # noqa
import numpy as np

include_dirs = [np.get_include(), "correlations"]

cython_extensions = [
    Extension(
        "correlations.utils", ["correlations/utils.pyx"], include_dirs=include_dirs
    )
]

setup(ext_modules=cythonize(cython_extensions))
