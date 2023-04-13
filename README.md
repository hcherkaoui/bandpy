[![pipeline status](https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/bandpy/badges/master/pipeline.svg)](https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/bandpy/-/commits/master)
[![coverage report](https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/bandpy/badges/master/coverage.svg)](https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/bandpy/-/commits/master)

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

    cd examples/
    python 1_demo_lin_ucb.py  # to play around with the examples


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
 * matrix_factorization

Dev
===

In order to launch the unit-tests, run the command::

    pytest  # run the unit-tests


In order to check the PEP 8 compliance level of the package, run the command::

    flake8 --count bandpy
