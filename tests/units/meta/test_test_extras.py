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

"""Tests for utilities shared across the test suite."""

import warnings

from tests.cases import VanguardTestCase


class OtherWarningSubclass(Warning):
    """Warning subclass used for testing."""


class SubclassOfUserWarning(UserWarning):
    """Warning subclass used for testing."""


class TestNotWarns(VanguardTestCase):
    def test_with_no_warnings(self):
        with self.assertNotWarns():
            pass

    def test_fails_with_warning(self):
        with self.assertRaises(self.failureException):
            with self.assertNotWarns():
                warnings.warn("A warning!", UserWarning)

    def test_passes_with_warning_not_of_given_type(self):
        with self.assertNotWarns(UserWarning):
            warnings.warn("A warning!", OtherWarningSubclass)

    def test_fails_with_warning_of_given_type(self):
        with self.assertRaises(self.failureException):
            with self.assertNotWarns(OtherWarningSubclass):
                warnings.warn("A warning!", OtherWarningSubclass)

    def test_fails_with_warning_of_descendant_of_given_type(self):
        with self.assertRaises(self.failureException):
            with self.assertNotWarns(UserWarning):
                warnings.warn("A warning!", SubclassOfUserWarning)
