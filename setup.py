from distutils.extension import Extension
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    "mciast_cython/iast/*.pyx"
]

setup(
    ext_modules = cythonize(extensions, annotate=True),
    include_dirs=[np.get_include()]
)