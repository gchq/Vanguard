"""
Code to test example notebooks.
"""
import os
import re
import unittest
from typing import Optional, Tuple, Any


from nbconvert.preprocessors import ExecutePreprocessor
import nbformat

_RE_SPHINX_EXPECT = re.compile("^# sphinx expect (.*Error)$")
TIMEOUT = 2400


class NotebookMetaClass(type):
    """
    A metaclass enabling dynamic tests to be rendered as real tests.

    Each notebook found in the 'examples' directory implies the creation of a
    specific test method, allowing for more verbose real-time feedback as
    opposed to subtests.
    """
    def __new__(mcs, name: str, bases: Optional[Tuple[Any]], namespace: Any):
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
                notebook_path = self.tests_to_notebook_paths[self._testMethodName]
                self._test_notebook(notebook_path)

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

    def _test_notebook(self, notebook_path: str) -> None:
        """No errors should be thrown."""
        with open(notebook_path) as rf:
            notebook = nbformat.read(rf, as_version=4)

        self.processor.preprocess(notebook)

        for cell_no, cell in enumerate(notebook.cells, start=1):
            if cell.cell_type == "code":
                self._verify_cell_outputs(cell_no, cell)

        if self.save_notebook_outputs:
            with open(notebook_path, "w") as wf:
                nbformat.write(notebook, wf, version=4)

    def _verify_cell_outputs(self, cell_no: int, cell) -> None:  # TODO: What is the expected type of cell? It must be a class that has the .outputs attribute, but is the class listed anywhere?
        for output in cell.outputs:
            if output.output_type == "error":
                self._verify_expected_errors(cell, cell_no, output)

    def _verify_expected_errors(self, cell, cell_no: int, output) -> None:
        """Verify if an error is expected in a cell."""
        cell_source_lines = cell.source.split("\n")
        match_if_cell_expected_to_ignore = _RE_SPHINX_EXPECT.match(cell_source_lines[0])
        if not match_if_cell_expected_to_ignore:
            self.fail(f"Should not have raised {output.ename} in cell number {cell_no}: "
                      f"{output.evalue}")
        else:
            expected_error = match_if_cell_expected_to_ignore.group(1)
            if output.ename != expected_error:
                self.fail(f"Expected {expected_error} in cell number {cell_no}, but {output.ename} was raised instead: "
                          f"{output.evalue}")
