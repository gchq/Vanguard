"""
Tests for the Vanguard package.

Vanguard's tests can be run from the command line, with numerous options:

python -m tests  # run all unit tests
python -m tests module  # attempt to run all unit tests in test_module.py
python -m tests sub1 sub2 module  # attempt to run all unit tests in sub1/sub2/test_module.py

Vanguard has a few different types of test:

python -m tests -u  # run all unit tests (default)
python -m tests -d  # run all doctests
python -m tests -e  # run all examples
python -m tests -a  # run all tests (u + d + e)

Other options are available and can be found by running the following:

python -m tests --help
"""
