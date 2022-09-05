#!/usr/bin/env python

"""
Ref: https://github.com/argoai/argoverse-api/blob/master/setup.py
A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from codecs import open  # To use a consistent encoding
from os import path

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gtsfm",
    version="0.2.0",
    description="",
    long_description=long_description,
    url="",
    author="",
    author_email="",
    license="APACHE",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="computer-vision",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">= 3.6",
    install_requires=[],
)
