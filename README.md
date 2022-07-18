[![pipeline status](https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/bandpy/badges/master/pipeline.svg)](https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/bandpy/-/commits/master)
[![coverage report](https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/bandpy/badges/master/coverage.svg)](https://rnd-gitlab-eu.huawei.com/Noahs-Ark/libraries/bandpy/-/commits/master)

Bandpy
======

Bandpy: multi-arms / multi-agents bandit Python package.

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
    jupyter-notebook  # to play around with the examples


Dependencies
============

The required dependencies to use the software are:

 * Matplotlib (>=3.0.0)
 * Numpy (>=1.10.0)

Dev
===

In order to launch the unit-tests, run the command::

    pytest  # run the unit-tests


In order to check the PEP 8 compliance level of the package, run the command::

    flake8 --count bandpy
