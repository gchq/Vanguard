"""
Utility functions for the configuration file.
"""
import itertools
import os
import shutil

import nbformat


def copy_filtered_files(source_folder, destination_folder, file_types=()):
    """Copy the contents of a folder across if they have a particular type."""
    for root, dirs, files in os.walk(source_folder):
        for dr in dirs:
            os.mkdir(os.path.join(root.replace(source_folder, destination_folder), dr))
        for file in files:
            if os.path.splitext(file)[1] in file_types:
                source_filename = os.path.join(root, file)
                dest_filename = source_filename.replace(source_folder, destination_folder)
                shutil.copyfile(source_filename, dest_filename)


def process_notebooks(notebook_file_paths):
    """Remove empty cells from Jupyter notebook files, filter sphinx commands and set raw mimetype."""
    for notebook_path in notebook_file_paths:
        if not notebook_path.endswith(".ipynb"):
            continue

        with open(notebook_path) as rf:
            notebook = nbformat.read(rf, as_version=4)

        notebook.cells = [
            cell  #
            for cell in notebook.cells
            if cell.source and not cell["source"].startswith("# sphinx ignore")
        ]

        correct_cell_numbers = itertools.count(1)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.execution_count = next(correct_cell_numbers)

        for cell in notebook.cells:
            if cell.source.startswith("# sphinx expect"):
                cell.source = "\n".join(cell.source.split("\n")[1:]).lstrip()

        for cell in notebook.cells:
            if cell["cell_type"] == "raw":
                cell["metadata"]["raw_mimetype"] = "text/restructuredtext"

        with open(notebook_path, "w") as wf:
            nbformat.write(notebook, wf, version=4)
