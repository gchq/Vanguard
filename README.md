# Vanguard: Advanced GPs

![version] ![python] ![coverage] ![pre-commit]

[version]: https://img.shields.io/badge/version-2.1.0-informational
[python]: https://img.shields.io/badge/python-3.8%7C3.9-informational
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

Vanguard's tests are contained in the `tests/` directory, and can be run with `unittest` or `pytest`.

Unit tests are in `tests/units`. There are two additional test files that dynamically run additional tests:
 - `test_doctests.py` finds and runs all doctests.
 - `test_examples.py` runs all notebooks under `examples/` as tests. These require `nbconvert` and `nbformat` to run,
and can take a significant amount of time to complete, so consider excluding `test_examples.py` from your test
discovery.

```shell
# Unittest:
$ python -m unittest discover -s tests/units # run unit tests
$ python -m unittest tests/test_doctests.py # run doctests
$ python -m unittest tests/test_examples.py # run example tests (slow)

# Pytest:
$ pytest tests/units # run unit tests
$ pytest tests/test_doctests.py # run doctests
$ pytest tests/test_examples.py # run example tests (slow)
```

Note that some tests are non-deterministic and as such may occasionally fail due to randomness.
Please try running them again before raising an issue.
