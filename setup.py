#!/usr/bin/env python

"""
Ref: https://github.com/argoai/argoverse-api/blob/master/setup.py
A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from codecs import open  # To use a consistent encoding
from pathlib import Path

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

# Get the long description from the README file
long_description = (Path(__file__).parent / "README.md").read_text()

setup(
    name="gtsfm",
    version="0.2.0",
    description="",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="",
    author="",
    author_email="",
    license="BSD-3-Clause",
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
