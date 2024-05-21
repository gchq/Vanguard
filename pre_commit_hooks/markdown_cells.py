"""
A simple pre-commit hook which forbids markdown cells in Jupyter notebooks.
"""
import argparse
from typing import Optional, Sequence

import nbformat


def count_markdown_cells_in_notebook(file_path: str, encoding="utf8") -> int:
    """Check a file and return the number of lambdas present."""
    with open(file_path, encoding=encoding) as rf:
        notebook = nbformat.read(rf, as_version=4)

    number_of_bad_cells = 0

    for cell_number, cell in enumerate(notebook.cells, start=1):
        if cell["cell_type"] == "markdown":
            number_of_bad_cells += 1
            source_preview = str(cell.source).replace("\n", " ")[:20]
            print(f"{file_path}:cell_{cell_number}: Markdown cell: '{source_preview}...'")

    return number_of_bad_cells


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*", help="Filenames to fix")
    args = parser.parse_args(argv)

    number_of_bad_cells = 0

    for filename in args.filenames:
        number_of_bad_cells += count_markdown_cells_in_notebook(filename)

    return int(bool(number_of_bad_cells))


if __name__ == "__main__":
    raise SystemExit(main())
