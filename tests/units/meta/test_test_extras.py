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

import pytest

from tests.cases import assert_not_warns


class OtherWarningSubclass(Warning):
    """Warning subclass used for testing."""


class SubclassOfUserWarning(UserWarning):
    """Warning subclass used for testing."""


class SecondSubclassOfUserWarning(UserWarning):
    """Warning subclass used for testing."""


class TestNotWarns:
    """Tests for the assert_not_warns() context manager."""

    def test_with_no_warnings(self):
        """Test that code that raises no warnings does not cause any error."""
        with assert_not_warns():
            pass

    def test_fails_with_warning(self):
        """Test that if a warning is raised, it is caught and raised as an error instead."""
        with pytest.raises(AssertionError):
            with assert_not_warns():
                warnings.warn("A warning!", UserWarning)

    def test_passes_with_warning_not_of_given_type(self):
        """Test that if a warning type is passed, warnings not of that type do not cause any error."""
        with pytest.warns(OtherWarningSubclass), assert_not_warns(UserWarning):
            warnings.warn("A warning!", OtherWarningSubclass)

    def test_fails_with_warning_of_given_type(self):
        """Test that if a warning type is passed, warnings of that type are caught and raised as an error instead."""
        with pytest.raises(AssertionError):
            with assert_not_warns(OtherWarningSubclass):
                warnings.warn("A warning!", OtherWarningSubclass)

    def test_fails_with_warning_of_descendant_of_given_type(self):
        """Test that if a warning type is passed, warnings of a subtype of that type are raised as an error instead."""
        with pytest.raises(AssertionError):
            with assert_not_warns(UserWarning):
                warnings.warn("A warning!", SubclassOfUserWarning)

    @pytest.mark.parametrize("warning_class", [SubclassOfUserWarning, SecondSubclassOfUserWarning])
    def test_fails_with_either_of_multiple_warnings(self, warning_class: type[Warning]):
        """Test usage of assert_not_warns with multiple warning types."""
        with pytest.raises(AssertionError):
            with assert_not_warns(SubclassOfUserWarning, SecondSubclassOfUserWarning):
                warnings.warn("A warning!", warning_class)

    def test_passes_with_warning_not_of_given_types(self):
        """Test that if a warning not of either of the given types is raised, it does not cause any error."""
        with pytest.warns(OtherWarningSubclass), assert_not_warns(SubclassOfUserWarning, SecondSubclassOfUserWarning):
            warnings.warn("A warning!", OtherWarningSubclass)

    def test_nested(self):
        """Test that nesting multiple copies of assert_not_warns works."""
        with pytest.raises(AssertionError):
            with assert_not_warns(OtherWarningSubclass), assert_not_warns(UserWarning):
                warnings.warn("A warning!", OtherWarningSubclass)
