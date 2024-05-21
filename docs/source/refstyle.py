"""
Contains a custom reference style for Vanguard.

See https://github.com/mcmtroffaes/sphinxcontrib-bibtex/blob/2.6.2/test/roots/test-bibliography_style_label_1/conf.py.
"""
# pylint: disable=import-error
from __future__ import annotations

from typing import Generator, Iterable

from pybtex.database import Entry
from pybtex.plugin import register_plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.labels import BaseLabelStyle

STYLE_NAME = "bib_key"


class MyLabelStyle(BaseLabelStyle):
    """
    A style which formats labels to use the original key in the .bib file.
    """

    def format_labels(self, sorted_entries: Iterable[Entry]) -> Generator[str, None, None]:
        for entry in sorted_entries:
            yield entry.key


class BibKeyStyle(UnsrtStyle):
    default_label_style = MyLabelStyle


register_plugin("pybtex.style.formatting", STYLE_NAME, BibKeyStyle)
