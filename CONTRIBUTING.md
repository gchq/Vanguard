# Contributing

If you would like to contribute to the development of Vanguard, you can do so in a number of ways:
- Highlight any bugs you encounter during usage, or any feature requests that would improve Vanguard by raising appropriate issues.
- Develop solutions to open issues and create pull requests (PRs) for the development team to review.
- Implement optimisations in the codebase to improve performance.
- Contribute example usages & documentation improvements.
- Increase awareness of Vanguard with other potential users.

All contributors must sign the [GCHQ Contributor Licence Agreement][cla - TODO].

The authors of Vanguard welcome any improvements that you may have, via a pull request.
Please familiarise yourself with this document before contributing.

We recommend installing our pre-commit hooks, which will be run automatically when a pull request is opened,
**regardless of whether you used them or not**. Save yourself some time, and run the following:

```shell
$ pip install pre-commit
$ pre-commit install
```

Before contributing, please get in touch with the Vanguard creators to check whether work is underway to address your concern,
and do not proceed until you have been given a JIRA ticket number.

## Pull requests

Currently, we are using a [GitHub Flow][github-flow] development approach.

- To avoid duplicate work, [search existing pull requests][gh-prs].
- All pull requests should relate to an existing issue.
  - If the pull request addresses something not currently covered by an issue, create a new issue first.
- Make changes on a [feature branch][git-feature-branch] instead of the main branch.
- Branch names should take one of the following forms:
  - `feature/<feature-name>`: for adding, removing or refactoring a feature.
  - `bugfix/<bug-name>`: for bug fixes.
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

## Code

Code must be documented, adequately tested and compliant with in style prior to merging into the main branch. To
facilitate code review, code should meet these standards prior to creating a pull request.

Some of the following points are checked by pre-commit hooks, although others require
manual implementation by authors and reviewers. Conversely, further style points that
are not documented here are enforced by pre-commit hooks; it is unlikely that authors
need to be aware of them.

### Style Guide

Vanguard is formatted using the [Ruff formatter][ruff], and is linted by both the [Ruff linter][ruff] and [Pylint][pylint].
Vanguard should be compatible with the supported Python versions listed in the README.
There should be very few inline comments; a combination of good docstrings and self-documenting code is preferred.
Docstrings should be written in a style similar to the [Sphinx format][sphinx-format].

Please note some Vanguard "specifics":

### Imports

Imports should be in alphabetical order within distinct groups: standard library imports, external imports, first-party (GCHQ) imports and relative (local) imports.
Each import should be on one line, unless multiple things are being imported.
This is enforced by pre-commit hooks.

```python
import argparse
import zlib

import numpy as np
from torch import Module, Tensor

from vanguard import kernels
```

### Docstrings

All docstrings must:
- Be written for private functions, methods and classes where their purpose or usage is not immediately obvious.
- Be written in [reStructured Text][sphinx-rst] ready to be compiled into documentation via [Sphinx][sphinx].
- Follow the [PEP 257][pep-257] style guide.
- Not have a blank line inserted after a function or method docstring unless the following statement is a function,
   method or class definition.
- Start with a capital letter unless referring to the name of an object, in which case match that case sensitively.
- Have a full stop at the end of the one-line descriptive sentence.
- Use full stops in extended paragraphs of text.
- Not have full stops at the end of parameter definitions.
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

:param a: Description of parameter a
:param b: Description of parameter b
:raises SyntaxError: Description of why a SyntaxError might be raised
:return: Description of return from function
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
Type comments should not be used, except for `#type: ignore`.
Don't use `from __future__ import annotations` - it can mask errors in the type annotations.

### Spelling and grammar

This project uses British English. Spelling is checked automatically by [cspell]. When a
word is missing from the dictionary, double check that it is a real word spelled
correctly. Contractions in object or reference names should be avoided unless the
meaning is obvious; consider inserting an underscore to effectively split into two
words. If you need to add a word to the dictionary, use the appropriate dictionary
inside the `.cspell` folder:

 - `acronyms.txt`: Acronyms or initialisms (self-explanatory).
 - `custom_misc.txt`: Things that don't fit in the other files, including exceptions for American English spellings where required
 - `library_terms.txt`: Terms from third-party code, such as package, function and argument names.
     Often these are shortenings or concatenations of English words, like `diag` or `lengthscale`.
     If cspell complains about a code term that is _not_ from third-party code, change the code term rather than adding
     it to the dictionary!
 - `maths_terms.txt`: Maths terms that are missing from the default dictionary, like "heteroskedastic".
 - `people.txt`: Names of people; largely these are authors cited in `references.bib`.

If the word fragment only makes sense as part of a longer phase, add the longer phrase
to avoid inadvertently permitting spelling errors elsewhere, e.g. add `Blu-Tack`
instead of `Blu`.

## Testing

Vanguard is subject to rigorous testing in both function and documentation.
Tests must be written in the `unittest` style, but can be run with either `unittest` or `pytest`.
These tests are automatically triggered by opening a pull request, and no merging can occur until they have all passed.

The `requirements.txt` file contains what is needed to run unit tests and doctests, but `docs/requirements-docs.txt` is required for the example tests.

Please ensure that tests are run regularly before opening a pull request, in order to catch errors early:

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

All examples notebooks should be run using **Python 3.8.3** with a kernel named "vanguard", and the format version should be **4.5**.
The pre-commit hooks will catch any discrepancies here.


## Documentation

Having complete documentation is important to easy usage of Vanguard.
Please ensure that any new code appears in the documentation, and is rendered correctly.
You should be warned of any broken internal links during the build process, and these will cause your pull request to be rejected.


## References

Vanguard is an academic project and therefore all work should be referenced.
New references should be placed in the file [`references.bib`](references.bib).
An entry with the keyword `Doe99` can then be referenced within a docstring anywhere with `[Doe99]_`.

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making
participation in our project, and our community a harassment-free experience for everyone.

### Our Standards

Examples of behaviour that contributes to creating a positive environment include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behaviour by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic address, without explicit permission
* Other conduct which could reasonably be considered inappropriate in a professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behaviour and are expected to take
appropriate and fair corrective action in response to any instances of unacceptable behaviour.

Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits,
issues, and other contributions that are not aligned to this Code of Conduct, or to ban temporarily or permanently any
contributor for other behaviors that they deem inappropriate, threatening, offensive, or harmful.

### Attribution

This Code of Conduct is adapted from version 1.4 of the [Contributor Covenant][contributor-covenant].

[contributor-covenant]: http://contributor-covenant.org/version/1/4/
[conventional_commits]: https://www.conventionalcommits.org
[cspell]: https://cspell.org/
[doctest]: https://docs.python.org/3/library/doctest.html
[gh-feature-request]: https://github.com/gchq/Vanguard/issues/new?assignees=&labels=enhancement%2Cnew&projects=&template=feature_request.yml&title=%5BFeature%5D%3A+
[gh-prs]: https://github.com/gchq/Vanguard/pulls?q=
[git-feature-branch]: https://www.atlassian.com/git/tutorials/comparing-workflows
[github-flow]: https://docs.github.com/en/get-started/quickstart/github-flow
[github-issues]: https://github.com/gchq/Vanguard/issues?q=
[pr-draft]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
[pr-ready]: https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request
[pylint]: https://www.pylint.org/
[ruff]: https://docs.astral.sh/ruff/
[run-tests]: https://github.com/gchq/Vanguard/actions/workflows/unittests.yml
[pep-257]: https://peps.python.org/pep-0257/
[sphinx]: https://www.sphinx-doc.org/en/master/index.html
[sphinx-format]: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html
[sphinx-rst]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html
