# © Crown Copyright GCHQ
#
# Licensed under the GNU General Public License, version 3 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to test example notebooks.
"""

import os
import re
import unittest
from typing import Any, Optional

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

_RE_SPHINX_EXPECT = re.compile("^# sphinx expect (.*Error)$")
TIMEOUT = 2400


class NotebookMetaClass(type):
    """
    A metaclass enabling dynamic tests to be rendered as real tests.

    Each notebook found in the 'examples' directory implies the creation of a
    specific test method, allowing for more verbose real-time feedback as
    opposed to subtests.
    """

    def __new__(mcs, name: str, bases: Optional[tuple[Any]], namespace: Any):
        """
        Prepare a class object.

        Creates a test function for each notebook in the examples folder, which will run that notebook and fail on
        any errors.

        :param name: The name of the class (i.e. the returned type's `__name__`).
        :param bases: The base classes of the returned class (i.e. the returned type's `__bases__`).
        :param namespace: The initial namespace of the returned class (i.e. the returned type's `__dict__`).
        """
        cls = super().__new__(mcs, name, bases, namespace)

        examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples", "notebooks")
        notebook_files = [entry for entry in os.listdir(examples_dir) if entry.endswith(".ipynb")]
        notebook_paths = (os.path.join(examples_dir, file) for file in notebook_files)
        test_names = (f"test_{file.split('.')[0]}_notebook" for file in notebook_files)

        cls.tests_to_notebook_paths = {test_name: full_path for test_name, full_path in zip(test_names, notebook_paths)}

        for test_name in cls.tests_to_notebook_paths:

            def inner_test(self) -> None:
                """
                Should not throw any errors.

                Notebook paths are established through a mapping in order to
                avoid unexpected behaviours which occur when using non-local
                loop variables within a new function.
                """
                notebook_path = self.tests_to_notebook_paths[self._testMethodName]  # pylint: disable=protected-access
                self._test_notebook(notebook_path)  # pylint: disable=protected-access

            inner_test.__name__ = test_name
            inner_test.__qualname__ = ".".join((cls.__qualname__, test_name))
            inner_test.__doc__ = "Should not throw any unexpected errors."
            setattr(cls, test_name, inner_test)

        return cls


class NotebookTests(unittest.TestCase, metaclass=NotebookMetaClass):
    """
    Tests that the notebooks can run properly.
    """

    def setUp(self) -> None:
        """Code to run before each test."""
        self.processor = ExecutePreprocessor(timeout=TIMEOUT, allow_errors=True)
        self.save_notebook_outputs = os.environ.get("SAVE_NOTEBOOK_OUTPUT", False)

    def _test_notebook(self, notebook_path: str, encoding: str = "utf8") -> None:
        """No errors should be thrown."""
        with open(notebook_path, encoding=encoding) as rf:
            notebook = nbformat.read(rf, as_version=4)

        self.processor.preprocess(notebook)

        for cell_no, cell in enumerate(notebook.cells, start=1):
            if cell.cell_type == "code":
                self._verify_cell_outputs(cell_no, cell)

        if self.save_notebook_outputs:
            with open(notebook_path, "w", encoding=encoding) as wf:
                nbformat.write(notebook, wf, version=4)

    def _verify_cell_outputs(self, cell_no: int, cell: nbformat.notebooknode.NotebookNode) -> None:
        for output in cell.outputs:
            if output.output_type == "error":
                self._verify_expected_errors(cell, cell_no, output)

    def _verify_expected_errors(
        self,
        cell: nbformat.notebooknode.NotebookNode,
        cell_no: int,
        output: nbformat.notebooknode.NotebookNode,
    ) -> None:
        """Verify if an error is expected in a cell."""
        cell_source_lines = cell.source.split("\n")
        match_if_cell_expected_to_ignore = _RE_SPHINX_EXPECT.match(cell_source_lines[1])
        if not match_if_cell_expected_to_ignore:
            if __debug__:
                print("Traceback:")
                for frame in output.traceback:
                    print(frame)
                self.fail(f"Should not have raised {output.ename} in cell number {cell_no}: {output.evalue}")
            else:
                self.fail(f"Got unexpected error in cell number {cell_no}")
        else:
            expected_error = match_if_cell_expected_to_ignore.group(1)
            if output.ename != expected_error:
                if __debug__:
                    print("Traceback:")
                    for frame in output.traceback:
                        print(frame)
                    self.fail(
                        f"Expected {expected_error} in cell number {cell_no}, but {output.ename} was raised instead: "
                        f"{output.evalue}"
                    )
                else:
                    self.fail(f"Got unexpected error in cell number {cell_no}")
