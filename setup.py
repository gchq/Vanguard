"""
Setup for the vanguard package.
"""
import os
import shutil

from setuptools import find_packages, setup

short_description = "Various easy-to-use extensions for Gaussian process models and a framework for" \
                    " composition of extensions."

with open("README.md", encoding="utf-8") as rf:
    long_description = "\n" + rf.read()


BIBTEX_FILE = "references.bib"
bibtex_destination_file = os.path.join(".", "vanguard", BIBTEX_FILE)

shutil.copyfile(BIBTEX_FILE, bibtex_destination_file)

setup(
    name="vanguard",
    version="2.1.0",
    description=short_description,
    long_description=long_description,
    author="GCHQ",
    license="GPLv3",
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    packages=find_packages(include=["vanguard", "vanguard.*"]),
    install_requires=[
        "bibtexparser",
        "gpytorch>=1.5.1",
        "matplotlib",
        "numpy>=1.20",
        "pandas",
        "scikit-learn",
        "scikit-learn-extra",
        "scipy",
        "torch>=1.7.0",
        "typing-extensions",
        "urllib3>=2.2.1"
    ],
    tests_require=[
        "pytest",
    ],
    extras_require={
        "dev": [
            "isort",
            "jupyterlab",
            "pre-commit",
            "pylint",
            "pyright",
            "pyroma",
            "ruff",
        ],
    },
    package_data={"vanguard": [BIBTEX_FILE]}
)

os.remove(bibtex_destination_file)
