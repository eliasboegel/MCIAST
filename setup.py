from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("mciast_cython/mciast_iast.pyx")
)