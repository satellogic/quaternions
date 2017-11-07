#!/usr/bin/env python3
from setuptools import setup


setup(
    name='satellogic_quaternions',
    version='0.1.3',
    author='Matias Graña',
    author_email='matias@satellogic.com',
    long_description='This is a library for dealing with quaternions in python in a unified way.',
    packages=['satellogic_quaternions'],
    license="Satellogic SA Copyright 2017. All our code is GPLv3 licensed.",
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
        'hypothesis'
    ],
)
