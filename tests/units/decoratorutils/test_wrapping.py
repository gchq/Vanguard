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

"""Tests for methods in `vanguard.decoratorutils.wrapping`."""

import unittest

from vanguard.decoratorutils import wraps_class


class TestWrapping(unittest.TestCase):
    def test_new_methods_are_not_wrapped(self):
        """Test that `wraps_class` only replaces attributes on methods that exist on the base class."""

        # We intentionally don't add a docstring here.
        class MyClass:
            def my_method_1(self):
                """Do absolutely nothing."""

        @wraps_class(MyClass)
        class MyInnerClass(MyClass):
            def my_method_1(self):
                """Do nothing. This docstring gets overwritten!"""

            def my_method_2(self):
                """Do even more nothing."""

        assert MyInnerClass.my_method_1.__doc__ == MyClass.my_method_1.__doc__
        assert MyInnerClass.my_method_2.__doc__ == """Do even more nothing."""
