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
        if "url" not in entry:
            if entry.get("archiveprefix") == "arXiv":
                try:
                    entry["url"] = "https://arxiv.org/abs/" + entry["eprint"]
                except KeyError as exc:
                    raise ValueError(f"Cannot calculate arXiv URL for {reference_id}: missing 'eprint'.") from exc
        else:
            entry["url"] = entry["url"].replace("\n", "")

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
