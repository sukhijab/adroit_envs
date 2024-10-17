#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'mujoco',
    'gymnasium',
    'numpy',
]

extras = {}
setup(
    name='adroit_envs',
    version='0.0.1',
    license="MIT",
    packages=find_packages(),
    python_requires='<= 3.12',
    package_data={
        'adroit_envs': ['assets/*'],  # Use relative path from the 'adroit_envs' package
    },
    include_package_data=True,
    install_requires=required,
    extras_require=extras,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)