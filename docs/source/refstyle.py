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
Contains a custom reference style for Vanguard.

See https://github.com/mcmtroffaes/sphinxcontrib-bibtex/blob/2.6.2/test/roots/test-bibliography_style_label_1/conf.py.
"""

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
