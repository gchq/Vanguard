# Vanguard: Advanced GPs

[![Unit Tests](https://github.com/gchq/Vanguard/actions/workflows/unittests.yml/badge.svg)](https://github.com/gchq/Vanguard/actions/workflows/unittests.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgchq%2Fvanguard-metadata%2Frefs%2Fheads%2Fmain%2Fcoverage%2Fcoverage_badge.json)](https://github.com/gchq/vanguard/actions/workflows/coverage.yml)
[![Pre-commit Checks](https://github.com/gchq/Vanguard/actions/workflows/pre_commit_checks.yml/badge.svg)](https://github.com/gchq/Vanguard/actions/workflows/pre_commit_checks.yml)
[![Linting: Pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Python version](https://img.shields.io/pypi/pyversions/vanguard-gp.svg)](https://pypi.org/project/vanguard-gp)
[![PyPI](https://img.shields.io/pypi/v/vanguard-gp)](https://pypi.org/project/vanguard-gp)
![Beta](https://img.shields.io/badge/pre--release-beta-red)

Vanguard is a high-level wrapper around [GPyTorch](https://gpytorch.ai/) and aims to provide a user-friendly interface for training and
using Gaussian process models. Vanguard's main objective is to make a variety of more advanced GP techniques in the
machine learning literature available for easy use by a non-specialists and specialists alike.
Vanguard is designed for modularity to facilitate straightforward combinations of different techniques.

Vanguard implements many advanced Gaussian process techniques, as showcased in our `examples` folder. These techniques
and others implemented within the Vanguard paradigm can be combined straightforwardly with minimal extra code, and
without requiring specialist GP knowledge.

## Installation

To install Vanguard:
```shell
pip install vanguard-gp
```
Note that it is `vanguard-gp` and not `vanguard`. However, to import the package, use
`from vanguard import ...`.

There are optional sets of additional dependencies:

* `vanguard-gp[test]` is required to run the tests;
* `vanguard-gp[doc]` is for compiling the Sphinx documentation;
* `vanguard-gp[notebook]` contains all dependencies for the example notebooks;
* `vanguard-gp[dev]` includes all tools and packages a developer of Vanguard might need.

Should the installation of Vanguard fail, you can see the versions used by the Vanguard
development team in `uv.lock`. You can transfer these to your own project as follows.
First, [install UV](https://docs.astral.sh/uv/getting-started/installation/). Then,
clone the repo from [GitHub](https://github.com/gchq/Vanguard). Next, run
```shell
uv export --format requirements-txt
```
which will generate a `requirements.txt`. Install this in your own project before trying
to install Vanguard itself,
```shell
pip install -r requirements.txt
pip install vanguard-gp
```

## Documentation

[Vanguard's documentation](https://vanguard.readthedocs.io/en/latest/) can be found online.

Alternatively, you can build the documentation from source - instructions for doing so can be found in
[`CONTRIBUTING.md`](CONTRIBUTING.md#documentation).

## Examples

Vanguard contains a number of example notebooks, contained in the `examples/notebooks` folder. These are designed to
showcase certain features of Vanguard within the context of a data science problem. To run them, you will need to first
install the additional requirements:

```shell
pip install vanguard-gp[notebook]
```

If you are in a virtual environment, you can then run the following to add the `vanguard` kernel to Jupyter, which makes
running the notebooks as frictionless as possible:

```shell
ipython kernel install --name vanguard --user
```

> **Warning**: Certain notebooks can take a long time to run, even on a GPU.  To see fully rendered examples, please visit the documentation.
