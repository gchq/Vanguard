# Contributing

If you would like to contribute to the development of Vanguard, you can do so in a number of ways:
- Highlight any bugs you encounter during usage, or any feature requests that would improve Vanguard by raising appropriate issues.
- Develop solutions to open issues and create pull requests (PRs) for the development team to review.
- Implement optimisations in the codebase to improve performance.
- Contribute example usages & documentation improvements.
- Increase awareness of Vanguard with other potential users.

All contributors must sign the [GCHQ Contributor Licence Agreement][cla].
(TODO: Update link when this is set up; see the linked issue.)

In addition, all contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).

The maintainers of Vanguard welcome any improvements that you may have, via a pull request.
Please familiarise yourself with this document before contributing.

We recommend installing our pre-commit hooks, which will be run automatically when a pull request is opened,
**regardless of whether you used them or not**. Save yourself some time, and run the following:

```shell
$ pip install pre-commit
$ pre-commit install
```

## Reporting issues

- [Search existing issues][github-issues] (both open **and** closed).
- Make sure you are using the latest version of Vanguard.
- Open a new issue:
  - For bugs use the [bug report issue template][gh-bug-report].
  - For features use the [feature request issue template][gh-feature-request].
  - This will make the issue a candidate for inclusion in future sprints, as-well as open to the community to address.
- If you are able to fix the bug or implement the feature, [create a pull request](#pull-requests) with the relevant changes.

## Getting set up
We use [uv] for dependency management. Install it via the shell as described in the linked documentation, or
simply with `pip install uv`.

Once uv is installed, simply run `uv sync` in the repo root to set up your virtual environment.

Once set up, you can use the virtual environment in a few ways:

 - Run commands in the virtual environment with `uv run <command>`; for example `uv run pytest tests/units`
 - Activate the virtual environment using the relevant activation script in `.venv/bin` (Linux)
    or `.venv/Scripts` (Windows)
 - (For IDEs) Set the Python executable in `.venv/bin` (Linux) or `.venv/Scripts` (Windows) as your active interpreter.

To add new packages as dependencies, simply add them to `pyproject.toml` and then run `uv sync` again.

For further details, see the [uv documentation][uv].

## Pull requests

We are using a [GitHub Flow][github-flow] development approach, where the trunk branch
is called `main`.

- To avoid duplicate work, [search existing pull requests][gh-prs].
- All pull requests should relate to an existing issue.
  - If the pull request addresses something not currently covered by an issue, create a new issue first.
- Make changes on a [feature branch][git-feature-branch] instead of the `main`
  branch.
- Branch names should take one of the following forms:
  - `feature/<feature-name>`: for adding, removing or refactoring a feature.
  - `fix/<bug-name>`: for bug fixes.
- Avoid changes to unrelated files in the same commit.
- Changes must conform to the [code](#code) guidelines.
- Changes must have sufficient [test coverage][run-tests].
- Delete your branch once it has been merged.

### Pull request process
- Create a [Draft pull request][pr-draft] while you are working on the changes to allow others to monitor progress and see the issue is being worked on.
- Pull in changes from upstream often to minimise merge conflicts.
- Make any required changes.
- Resolve any conflicts with the target branch.
- [Change your PR to ready][pr-ready] when the PR is ready for review. You can convert back to Draft at any time.

Do **not** add labels like `[RFC]` or `[WIP]` to the title of your PR to indicate its state.
Non-Draft PRs are assumed to be open for comments; if you want feedback from specific people, `@`-mention them in a comment.

### Pull request commenting process
- Use a comment thread for each required change.
- Reviewer closes the thread once the comment has been resolved.
- Only the reviewer may mark a thread they opened as resolved.

### Commit messages

Follow the [conventional commits guidelines][conventional_commits] to *make reviews easier* and to make the git logs more valuable.
An example commit, including reference to some GitHub issue #123, might take the form:

```
feat: add gpu support for matrix multiplication

If a gpu is available on the system, it is automatically used when performing matrix
multiplication within the code.

BREAKING CHANGE: numpy 1.0.2 no longer supported

Refs: #123
```

### Breaking changes and deprecation

Since we are still pre-1.0, [SemVer] states that any release may contain breaking
changes. However, breaking changes should not be made without warning.

Any breaking changes must have a deprecation period of at least **one minor release,
or one month (whichever is longer),** before the breaking change is made. If the change
is one that may require significant changes to client code, such as removing a function
or class entirely, the deprecation period must instead be at least **two minor releases,
or two months (whichever is longer).**

Ensure that during the deprecation period, the old behaviour still works, but raises a
`DeprecationWarning` with an appropriate message. If at all possible, ensure that there
is straightforward signposting for how users should change their code to use
non-deprecated parts of the codebase instead.

As an example, this is what the deprecation period for renaming `my_old_function` to
`my_new_function` would look like ([docstrings](#docstrings) have been omitted for
brevity, but should be included in real code!):

```python
# v0.1.0:
def my_old_function(x: int) -> int:
    return x + x + x + x

# v0.2.0:
def my_new_function(x: int) -> int:
    return x*4

@deprecated("Renamed to my_new_function; will be removed in v0.3.0")
def my_old_function(x: int) -> int:
    return my_new_function(x)

# v0.3.0:
def my_new_function(x: int) -> int:
    return x*4
```

## Code

Code must be documented, adequately tested and compliant with our [style guide](#style-guide) prior to merging
into the `main` branch.
To facilitate code review, code should meet these standards prior to creating a pull request.

Some of the following points are checked by pre-commit hooks, although others require
manual implementation by authors and reviewers. Conversely, further style points that
are not documented here are enforced by pre-commit hooks; it is unlikely that authors
need to be aware of them.

## Style Guide

 - Vanguard is formatted using the [Ruff formatter][ruff], and is linted by both the [Ruff linter][ruff] and [Pylint][pylint].
 - Vanguard should be compatible with the supported Python versions listed in the README.
 - Avoid inline comments; a combination of good docstrings and self-documenting code is preferred.
 - Follow [PEP8][pep-8] style where possible.
 - Use clear naming of variables rather than mathematical shorthand (e.g. `kernel` instead of `k`).
 - Type annotations must be used for all function or method parameters.

Please note some Vanguard "specifics":

### Docstrings

All docstrings must:
- Be written for private functions, methods and classes where their purpose or usage is not immediately obvious.
- Be written in [reStructured Text][sphinx-rst] ready to be compiled into documentation via [Sphinx][sphinx].
- Follow the [PEP 257][pep-257] style guide.
- Start with a capital letter unless referring to the name of an object, in which case match that case sensitively.
- Have a full stop at the end of the one-line descriptive sentence.
- Use full stops in extended paragraphs of text.
- Have full stops at the end of parameter definitions.
- Not use type hints in `:param:` fields - e.g. use `:param x:` rather than `:param float x:`. Use function annotations
   instead.
- Not use `:type:` or `:rtype:` fields. Use function annotations instead.
- Contain [references](#references) where necessary.

Class docstrings must contain at least 3 lines, unless the class is an internal class used inside a decorator, or
a stub class used within unit tests. This should consist of a brief description, a longer description
potentially making use of references, and ideally an example.

Additionally,
- If a `:param:` or similar line requires more than the max line length, use multiple lines. Lines after the first should
   be indented by a further 4 spaces.
- Class `__init__` methods should not have docstrings. All constructor parameters should be listed at the end of the class
  docstring. `__init__` docstrings will not be rendered by Sphinx. Any developer comments should be contained in a regular
  comment.
- Any examples in docstrings should be written to be compatible with [doctest], and must pass testing.


Each docstring for a public object should take the following structure:
```python
"""
Write a one-line descriptive sentence as an active command.

As many paragraphs as is required to document the object.

:Example:
    >>> string_one = "Add some brief but informative examples"
    >>> string_two = "they'll be automatically tested with doctest!"
    >>> f"{string_one} - {string_two}"
    "Add some brief but informative examples - they'll be automatically tested with doctest!"

:param a: Description of parameter a.
:param b: Description of parameter b.
:raises SyntaxError: Description of why a SyntaxError might be raised.
:return: Description of return from function.
"""
```
If the function does not return anything, the return line above can be omitted.

It is also helpful to use *notes* and *warnings* in the docstrings, which will draw attention in the documentation - for example:
```python
def which_day() -> str:
    """
    Return the current day of the week.

    .. warning::
        This function will only be correct on a Friday.
    """
    return "Friday"
```

There are numerous other things which can be included in docstrings, such as equations and even plots.
Please refer to other examples in Vanguard to see how these work.

### Module docstrings

Each module should have a docstring.
If intended to be included in the documentation, then this should be descriptive.
The presence of module docstrings is enforced by the pre-commit hooks.
For example:

```python
"""
This new Gaussian process technique will change your life.
"""
```

If the module documentation is *not* intended to be included in the compiled documentation - for example, at the top
of a unit test file - then it is sufficient to simply describe the contents of the file:

```python
"""
Tests for the MagicTechnique decorator.
"""
```

### Comments

Comments must:
- Start with a capital letter unless referring to the name of an object, in which case match that case sensitively.
- Not end in a full stop for single-line comments in code.
- End with a full stop for multi-line comments.

### Maths overflow

Prioritise overfull lines for mathematical expressions over artificially splitting them into multiple equations in
both comments and docstrings.

### Thousands separators

For hardcoded integers >= 1000, an underscore should be written to separate the thousands, e.g. 10_000 instead of 10000.

### Type annotations

All functions and methods should have type annotations for all parameters and for the return type.
Type comments should not be used, _including_ `# type: ignore`.
If you're sure the code's typing is correct, but Pyright still raises errors, use a specific
`# pyright: ignore [specific-error]`, with an accompanying comment explaining why the ignore is necessary.
Don't use `from __future__ import annotations` - it can mask errors in the type annotations.

### Spelling and grammar

This project uses British English. Spelling is checked automatically by [cspell]. When a
word is missing from the dictionary, double check that it is a real word spelled
correctly. Contractions in object or reference names should be avoided unless the
meaning is obvious; consider inserting an underscore to effectively split into two
words. If you need to add a word to the dictionary, use the appropriate dictionary
inside the `.cspell` folder - see [`.cspell/README.md`](.cspell/README.md) for an explanation of
which dictionary to use for what.

If the word fragment only makes sense as part of a longer phase, add the longer phrase
to avoid inadvertently permitting spelling errors elsewhere, e.g. add `Blu-Tack`
instead of `Blu`.

## Testing

Vanguard is subject to rigorous testing in both function and documentation.

Either [Pytest][pytest] or [Unittest][unittest] can be used to write tests for Vanguard.
[Pytest][pytest] is recommended where it would simplify code, such as for parameterized tests.
As much effort should be put into developing tests as is put into developing the code.
Tests should be provided to test functionality and also ensuring exceptions and warnings are raised or
managed appropriately. This includes:
- Unit testing of new functions added to the codebase
- Verifying all existing tests pass with the integrated changes

Keep in mind the impact on runtime when writing your tests. Favour more tests that are smaller rather than a few large
tests with many assert statements unless it would significantly affect run time, e.g. due to excess set up or duplicated
function calls.

The test suite is run automatically on each opened pull request, and no merging can occur until the whole suite has
passed.

The `requirements.txt` file contains what is needed to run unit tests and doctests,
but `docs/requirements-docs.txt` is required for the example tests.

Please ensure that tests are run regularly _before_ opening a pull request, in order to catch errors early:

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

### Testing before releases to PyPI

Before a release is issued on PyPI, all tests for Vanguard will be run on a GPU machine.
This avoids having to incorporate GPU runners into the CI/CD, but still tests issues that can arise with torch and gpytorch.
However, note that code pushed to `main` may not necessarily have been tested on a
GPU machine until a release to PyPI is made.
If you observe any issues on GPU machines using the code, please raise an issue detailing the behaviour, and create a PR with the relevant fix if possible.

## Examples

Writing an example notebook for new functionality is encouraged, as it improves understanding of a technique.
Any references to the code should be documented similarly to docstrings, which often requires cells to be in "raw" mode.
Please make use of the `# sphinx ignore` comment in cells which are required to run the notebook, but are not useful for the example.

Notebooks should follow the same general format:

1. **Data:** A brief explanation of the data, and why it is used.
   New datasets should be added to the `vanguard.datasets` subpackage for ease, as they can then be referenced from future examples.
2. **Modelling:** A section showing the implementation, training and results of your new technique.
   Code should be as brief and clear as possible, and is subject to the same style guide as the rest of Vanguard.
   Please keep line length to a minimum to avoid readability issues.
3. **Conclusions** Summarise and interpret the results.
   These do not need to break any records, as the majority of the example notebooks are there for exposition purposes.

Please minimise functional code within your notebook.
For example, functions for plotting datasets fit better as methods of the corresponding `Dataset` subclass (which you should have added with a new dataset).
If you need a new plotting function, consider adding a new, more specific plotting method, or attempting to generalise the existing one.


## Documentation

Having complete documentation is important to easy usage of Vanguard.
Please ensure that any new code appears in the documentation, and is rendered correctly.
You should be warned of any broken internal links during the build process, and these will cause your pull request to be rejected.

### Documentation dependencies

Building the documentation yourself requires the additional documentation dependencies. Install these with

```shell
pip install -e .[doc]
```

from the repo root. Alternatively, we maintain a set of pinned dependencies that should allow the documentation to
be built without error on Linux - to install these, run

```shell
pip install -r requirements-docs.txt --no-deps
```

from the repo root. Note that the pinned dependencies are not guaranteed to work on Windows or macOS.

In both cases, these commands should be run in a fresh virtual environment (to avoid any issues with already-existing
packages), and on Python 3.12.

### Building the documentation

To build the documentation, run

```shell
python -m sphinx -b html -WEan --keep-going docs/source docs/build
```

from the repo root. This is the same command that is used in the CI pipeline to check that the documentation builds
correctly, so if it builds without error in an isolated environment containing only the required dependencies, it should
also build without error in the CI pipeline.

After building, you should then run the external link checker with

```shell
python -m sphinx -b linkcheck docs/source docs/build
```

## Releases

Releases are made on an ad-hoc basis. When the maintainers decide the codebase is ready
for another release:

1. Create an issue for the release.
2. Run additional release tests on `main` including GPU testing, as described in
   [Testing before releases to PyPI](#Testing-before-releases-to-PyPI).
3. Fix any issues and merge into `main`, iterating until we have a commit on
   `main` that is ready for release, except for housekeeping that does not affect the
   functionality of the code.
4. Create a branch `release/#.#.#` off the identified commit, populating with the target
   version number.
5. Tidy `CHANGELOG.md` including:
   - Move the content under `Unreleased` to a section under the target version number.
   - Create a new unpopulated `Unreleased` section at the top.
   - Update the hyperlinks to Git diffs at the bottom of the file so that they compare
     the relevant versions.
6. Update the version number in `vanguard/__init.py__`.
7. Create and review a pull request.
8. Once approved, create a release in GitHub pointing at the final commit on the release
   branch.
9. Build and publish to PyPI and ReadTheDocs.
10. Merge the release branch into `main`.

## References

Vanguard is an academic project and therefore all work should be referenced.
New references should be placed in the file [`references.bib`](references.bib).
An entry with the keyword `Doe99` can then be referenced within a docstring anywhere with ``:cite:`Doe99` ``.

[cla]: https://github.com/gchq/Vanguard/issues/230
[conventional_commits]: https://www.conventionalcommits.org
[cspell]: https://cspell.org/
[doctest]: https://docs.python.org/3/library/doctest.html
[gh-bug-report]: https://github.com/gchq/Vanguard/issues/new?assignees=&labels=new%2Cbug&projects=gchq%2F16&template=bug_report.yml
[gh-feature-request]: https://github.com/gchq/Vanguard/issues/new?assignees=&labels=new%2Cenhancement&projects=gchq%2F16&template=feature_request.yml
[gh-prs]: https://github.com/gchq/Vanguard/pulls
[git-feature-branch]: https://www.atlassian.com/git/tutorials/comparing-workflows
[github-flow]: https://docs.github.com/en/get-started/quickstart/github-flow
[github-issues]: https://github.com/gchq/Vanguard/issues?q=
[pep-8]: https://peps.python.org/pep-0008/
[pep-257]: https://peps.python.org/pep-0257/
[pr-draft]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
[pr-ready]: https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request
[pylint]: https://www.pylint.org/
[pytest]: https://docs.pytest.org/
[ruff]: https://docs.astral.sh/ruff/
[run-tests]: https://github.com/gchq/Vanguard/actions/workflows/unittests.yml
[semver]: https://semver.org/
[sphinx]: https://www.sphinx-doc.org/en/master/index.html
[sphinx-format]: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html
[sphinx-rst]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html
[unittest]: https://docs.python.org/3/library/unittest.html
[uv]: https://docs.astral.sh/uv/
