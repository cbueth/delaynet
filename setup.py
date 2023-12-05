"""Module setup."""
from setuptools import find_packages, setup

setup(
    name="delaynet",
    packages=find_packages(),
    description="Delay Propagation in Transportation Networks",
    author="Carlson Büth",
    license="BSD-3-Clause",
    install_requires=["numpy", "scipy", "statsmodels", "scikit-learn", "numba", "mkl"],
    extras_require={"dev": ["pytest", "pytest-cov", "coverage"]},
)
