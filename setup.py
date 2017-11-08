#!/usr/bin/env python3
import os.path
from setuptools import setup, find_packages


# https://packaging.python.org/guides/single-sourcing-package-version/
version = {}
with open(os.path.join("quaternions", "version.py")) as fp:
    exec(fp.read(), version)


setup(
    name='satellogic_quaternions',
    version=version["__version__"],
    author='Matias Graña',
    author_email='matias@satellogic.com',
    long_description='This is a library for dealing with quaternions in python in a unified way.',
    packages=find_packages(exclude=["tests"]),
    license="GPLv3",
    classifiers=[
        'Development Status :: 1 - Beta',
        'Environment :: Console',
        'Intended Audience :: Satellites',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],
    install_requires=[
        'numpy',
    ],
    extras_require={
        "dev": [
            "hypothesis",
        ]
    }
)
