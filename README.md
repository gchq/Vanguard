# Vanguard: Advanced GPs

![version] ![python] ![coverage] ![pre-commit] ![Beta]

[Beta]: https://img.shields.io/badge/pre--release-beta-red
[version]: https://img.shields.io/badge/version-2.1.0-informational
[python]: https://img.shields.io/badge/python-3.9--3.12-informational
[coverage]: https://img.shields.io/badge/coverage-91%25-brightgreen
[pre-commit]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=orange

Vanguard is a high-level wrapper around [GPyTorch](https://gpytorch.ai/) and aims to provide a user-friendly interface for training and using Gaussian process models.
Vanguard's main objective is to make a variety of more advanced GP techniques in the machine learning literature available for easy use by a non-specialists and specialists alike.
Vanguard is designed for modularity to facilitate straightforward combinations of different techniques.

Vanguard implements many advanced Gaussian process techniques, as showcased in our `examples` folder. These techniques and others implemented within the Vanguard paradigm can be combined straightforwardly with minimal extra code, and without requiring specialist GP knowledge.

## Installation

To install Vanguard:
```shell
$ pip install vanguard-gp
```
Note that it is `vanguard-gp` and not `vanguard`. However, to import the package, use
`from vanguard import ...`.

If the code is not running properly, recreate the environment with `pip install -r requirements.txt --no-deps`.

## Documentation

Vanguard's documentation can be found online at [TODO: LINK GOES HERE].

Alternatively, you can build the documentation from source - instructions for doing so can be found in
[`CONTRIBUTING.md`](CONTRIBUTING.md#documentation).

## Tests

Vanguard's tests are contained in the `tests/` directory, and can be run with `pytest`. The tests are arranged
as follows:
 - `tests/units` contains unit tests. These should be fairly quick to run.
 - `tests/integration` contains integration tests, which may take longer to run.
 - `tests/test_doctests.py` finds and runs all doctests. This should be fairly quick to run.
 - `tests/test_examples.py` runs all notebooks under `examples/` as tests. These require `nbconvert` and `nbformat` to run,
and can take a significant amount of time to complete, so consider excluding `test_examples.py` from your test
discovery.


```shell
$ pytest # run all tests (slow)
$ pytest tests/units # run unit tests
$ pytest tests/integration # run integration tests (slow)
$ pytest tests/test_doctests.py # run doctests
$ pytest tests/test_examples.py # run example tests (slow)
```

Our PR workflows run our tests with the `pytest-beartype` plugin. This is a runtime type checker that ensures all
our type hints are correct. In order to run with these checks locally, add
`--beartype-packages="vanguard" -m "not no_beartype"` to your pytest invocation. You should then separately run pytest
with `-m no_beartype` to ensure that all tests are run. The reason for this separation is that some of our tests check
that our handling of inputs of invalid type are correct, but `beartype` catches these errors before we get a chance to
look at them, causing the tests to fail; thus, these tests need to be run separately _without_ beartype.

Since different Python versions have different versions of standard library and third-party modules, we can't guarantee
that type hints are 100% correct on all Python versions. Type hints are only tested for correctness on the latest
version of Python (3.12).

For example, to run the unit tests with type checking:

```shell
$ pytest tests/units --beartype-packages="vanguard" -m "not no_beartype"  # run unit tests with type checking
$ pytest tests/units -m no_beartype  # run unit tests that are incompatible with beartype
```
