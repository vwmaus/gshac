"""Build script for the GSHAC C extension."""
from setuptools import setup, Extension
import numpy as np

ext = Extension(
    'gshac._gshac',
    sources=['src/gshac/_gshac.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=['-O3', '-march=native', '-Wall'],
)

setup(ext_modules=[ext])
