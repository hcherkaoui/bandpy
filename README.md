<p align="center">
<a href="https://www.python.org/download/releases/3.0/"><img alt="Python Version" src="https://img.shields.io/badge/python-3-blue.svg"></a>
<a href="https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/bandpy/-/commits/master"><img alt="Pipeline Status" src="https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/bandpy/badges/master/pipeline.svg"></a>
<a href="https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/bandpy/-/commits/master"><img alt="Coverage Status" src="https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/bandpy/badges/master/coverage.svg"></a>
<a href="https://opensource.org/licenses/BSD-3-Clause"><img alt="License: BSD-3" src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Bandpy
======


Short description:
------------------
Multi-armed single/multi-agent bandits Python package.

Description:
------------
Bandpy aims to provide all classical agents and controllers on a
various synthetic and real data environnments to ease benchmark for research
and R&D purposes.

Important links
===============

- Official source code repo: https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/bandpy

Installation
============

In order install the package, run::

    pip install -r requirements.txt
    pip install -e .


To validate the installation, run::

    cd examples/1_illustrations
    python 0_demo_lin_ucb.py  # to play around with the examples


Dependencies
============

The required dependencies to use the software are:

 * Matplotlib (>=3.0.0)
 * Numpy (>=1.10.0)
 * Pandas (>=1.4.1)
 * Scipy (>=1.8.0)
 * Joblib (>=0.16.0)
 * Scikit-Learn (>=1.0.2)
 * Networkx (>=2.8.6)
 * Gym (>=0.23.1)
 * Wget (>=3.0)
 * matrix_factorization

Dev
===

In order to launch the unit-tests, run the command::

    pytest  # run the unit-tests


In order to check the PEP 8 compliance level of the package, run the command::

    flake8 --ignore=E501 --count bandpy
