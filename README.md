# Vanguard: Advanced GPs

![version] ![python] ![coverage] ![pre-commit]

[version]: https://img.shields.io/badge/version-2.1.0-informational
[python]: https://img.shields.io/badge/python-3.7%7C3.8%7C3.9-informational
[coverage]: https://img.shields.io/badge/coverage-91%25-brightgreen
[pre-commit]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=orange

Vanguard is a high-level wrapper around [GPyTorch](https://gpytorch.ai/) and aims to provide a user-friendly interface for training and using Gaussian process models.
Vanguard's main objective is to make a variety of more advanced GP techniques in the machine learning literature available for easy use by a non-specialists and specialists alike.
Vanguard is designed for modularity to facilitate straightforward combinations of different techniques.

Vanguard implements many advanced Gaussian process techniques, as showcased in our `examples` folder. These techniques and others implemented within the Vanguard paradigm can be combined straightforwardly with minimal extra code, and without requiring specialist GP knowledge.

## Installation

Vanguard can be installed directly from source:

```shell
$ pip install .
```

If the code is not running properly, recreate the environment with `pip install -r requirements.txt --no-deps`.

## Tests

Vanguard's tests can be run from the command line, with numerous options:

```shell
$ python3 -m tests  # run all unit tests
$ python3 -m tests module  # attempt to run all unit tests in test_module.py
$ python3 -m tests sub1 sub2 module  # attempt to run all unit tests in sub1/sub2/test_module.py
```

Vanguard has a few different types of test:

```shell
$ python3 -m tests -u  # run all unit tests (default)
$ python3 -m tests -d  # run all doctests
$ python3 -m tests -e  # run all examples (additional dependencies are required, see below)
$ python3 -m tests -a  # run all tests (u + d + e)
```

Other options are available and can be found by running the following:

```shell
$ python3 -m tests --help
```

* Some tests are non-deterministic and as such may occasionally fail due to randomness. Please try running them again before raising an issue.
* Example tests require `nbconvert` and `nbformat` to run. The tests will only check if the notebooks run, and not how they perform. They are often very slow to complete.
