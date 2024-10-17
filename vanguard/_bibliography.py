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
Contains functions for importing the bibliography from a .bib file.
"""

import os
import sys

import bibtexparser


def _import_bibliography(bibtex_file_path, encoding="utf8"):
    """Import a .bib file as a dictionary."""
    with open(bibtex_file_path, encoding=encoding) as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    references = {}

    for entry in bib_database.entries:
        reference_id = entry.pop("ID")

        # Try and get a URL, if possible.
        if "url" in entry:
            # Either get one listed directly in the entry...
            entry["url"] = entry["url"].replace("\n", "")
        elif entry.get("archiveprefix") == "arXiv":
            # ...or for arXiv specifically, we can infer the URL from the "eprint" field.
            try:
                entry["url"] = "https://arxiv.org/abs/" + entry["eprint"]
            except KeyError as exc:
                # Only print the full reference_id if running without optimisations - printing unsanitised user inputs
                # to the console presents a security risk.
                if __debug__:
                    raise ValueError(f"Cannot calculate arXiv URL for {reference_id}: missing 'eprint'.") from exc
                else:
                    raise ValueError("Cannot calculate arXiv URL: missing 'eprint'.") from exc

        references[reference_id] = entry

    return references


def find_bibliography(file_name="references.bib"):
    """Find the bibliography from the references file if it exists."""
    vanguard_dir = os.path.dirname(__file__)
    vanguard_parent_dir = os.path.dirname(vanguard_dir)
    bibliography_path = [vanguard_dir, vanguard_parent_dir] + sys.path
    for path in bibliography_path:
        try:
            bibliography = _import_bibliography(os.path.join(path, file_name))
        except OSError:
            pass
        else:
            return bibliography
    return None
