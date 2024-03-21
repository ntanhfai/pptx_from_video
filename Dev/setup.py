# setup.py

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

sourceFiles = ["libcalculator/square.py"]

extensions = cythonize(Extension(
    name="libcalculator.libsquarepython",
    sources=sourceFiles
))

kwargs = {
    "name": "libcalculator",
    "packages": find_packages(),
    "ext_modules": extensions
}

setup(**kwargs)
