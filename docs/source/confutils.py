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
Utility functions for the configuration file.
"""

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


def process_notebooks(notebook_file_paths, notebooks_to_skip, encoding="utf8"):
    """
    Remove empty cells from Jupyter notebook files, filter sphinx commands and set raw mimetype.

    :param notebook_file_paths: Paths to notebooks that will be loaded and adjusted.
    :param notebooks_to_skip: List of notebooks to skip running during sphinx compile, e.g. due
        to long run times.
    :param encoding: Encoding to use when reading notebook files.
    """
    for notebook_path in notebook_file_paths:
        if not notebook_path.endswith(".ipynb"):
            continue

        with open(notebook_path, encoding=encoding) as rf:
            notebook = nbformat.read(rf, as_version=4)

        notebook.cells = [
            cell  #
            for cell in notebook.cells
            if cell.source and not cell["source"].startswith("# sphinx ignore")
        ]

        # Apply any specific exclusions we want
        for cell in notebook.cells:
            if cell.source.startswith("# sphinx expect"):
                cell.source = "\n".join(cell.source.split("\n")[1:]).lstrip()

        # Ensure nbsphinx correctly compiles text cells
        for cell in notebook.cells:
            if cell["cell_type"] == "raw":
                cell["metadata"]["raw_mimetype"] = "text/restructuredtext"

        # Skip long-running notebooks
        if any(exclude_string in notebook_path for exclude_string in notebooks_to_skip):
            notebook["metadata"]["nbsphinx"] = {"execute": "never"}

        with open(notebook_path, "w", encoding=encoding) as wf:
            nbformat.write(notebook, wf, version=4)
