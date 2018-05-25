# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
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


# with open('Readme.md') as f:
#     readme = f.read()

# with open('LICENSE') as f:
#     license = f.read()

setup(
    name='correlations',
    version='0.0.1',
    description='Compute correlations functions for Gaussian Random Fields.',
    classifiers=[
        'Development status :: 1 - Alpha',
        'License :: CC-By-SA2.0',
        'Programming Language :: Python',
        'Topic :: Gaussian Random Field'
    ],
    author='Corentin Cadiou',
    author_email='contact@cphyc.me',
    packages=['correlations'],
    package_dir={'correlations': 'correlations'},
    package_data={'correlations': [
        'data/power.dat',
        'correlations/*.pyx'
    ]},
    install_requires=[
        'numpy',
    ],
    include_package_data=True,
    ext_modules=cythonize(cython_extensions)
)
