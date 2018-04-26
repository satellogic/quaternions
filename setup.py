#!/usr/bin/env python3
from setuptools import setup, find_packages
import versioneer


setup(
    name='satellogic_quaternions',
    version=versioneer.get_version(),
    author='Matias Graña, Enrique Toomey, Slava Kerner',
    author_email='matias@satellogic.com',
    description='This is a library for dealing with quaternions in Python in a unified way.',
    long_description=open('README.md').read(),
    packages=find_packages(exclude=["tests"]),
    license="GPLv3",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],
    install_requires=[
        'numpy',
    ],
    extras_require={
        'dev': [
            'flake8 >= 2.5.4',
            'hypothesis',
            'pytest',
            'pytest-coverage',
        ]
    },
    cmdclass=versioneer.get_cmdclass(),
)
