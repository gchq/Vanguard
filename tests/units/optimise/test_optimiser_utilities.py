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

"""Tests for the utility classes in optimiser.py."""

import unittest
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor
from torch.nn import Module

from tests.cases import get_default_rng
from vanguard.optimise.optimiser import MaxLengthHeapQ, Parameters


class TestMaxLengthHeapQ(unittest.TestCase):
    """Test the `MaxLengthHeapQ` class."""

    def test_push_within_max_size(self):
        """Test that if we push fewer items than the max size, they all stay in the heap."""
        heap = MaxLengthHeapQ(100)

        for x in range(10):
            heap.push(x)
        for x in range(10):
            assert x in heap

    def test_push_over_max_size(self):
        """Test that if we push more items than the max size, the smallest are removed from the heap."""
        heap = MaxLengthHeapQ(5)
        for x in range(10):
            heap.push(x)

        for x in range(5):
            assert x not in heap

        for x in range(5, 10):
            assert x in heap

    def test_nlargest(self):
        """Test that `nlargest(n)` returns the n largest values."""
        items = [x for x in range(10)]
        get_default_rng().shuffle(items)

        heap = MaxLengthHeapQ(100)
        for x in items:
            heap.push(x)

        assert set(heap.nlargest(3)) == {x for x in range(10 - 3, 10)}

    def test_best(self):
        """Test that `best()` returns the single largest value."""
        items = [x for x in range(10)]
        get_default_rng().shuffle(items)

        heap = MaxLengthHeapQ(100)
        for x in items:
            heap.push(x)

        assert heap.best() == 9


class TestParameters(unittest.TestCase):
    """Tests for the `Parameters` class."""

    def test_parameters_unequal(self):
        """Test the comparison operators on `Parameters` instances of unequal value."""
        a = Parameters({}, 1)
        b = Parameters({}, 2)

        assert a < b
        assert a <= b
        assert a != b
        assert b >= a
        assert b > a
        # Ignore all "unnecessary negation" errors - we need to check that these methods actually return False
        # as expected, so we don't have something weird like `a < b and not b > a`!
        assert not a > b  # pylint: disable=unnecessary-negation
        assert not a >= b  # pylint: disable=unnecessary-negation
        assert not a == b  # pylint: disable=unnecessary-negation
        assert not b <= a  # pylint: disable=unnecessary-negation
        assert not b < a  # pylint: disable=unnecessary-negation

    def test_parameters_equal(self):
        """Test the comparison operators on `Parameters` instances of equal value."""
        a = Parameters({}, 1)
        b = Parameters({}, 1)

        assert a == b
        assert a <= b
        assert a >= b
        assert b <= a
        assert b >= a
        # Ignore all "unnecessary negation" errors - we need to check that these methods actually return False
        # as expected, so we don't have something weird like `a < b and not b > a`!
        assert not a != b  # pylint: disable=unnecessary-negation
        assert not a < b  # pylint: disable=unnecessary-negation
        assert not a > b  # pylint: disable=unnecessary-negation
        assert not b < a  # pylint: disable=unnecessary-negation
        assert not b > a  # pylint: disable=unnecessary-negation

    def test_parameter_type_error_comparison(self):
        """Test that comparison with something other than a `Parameters` instance raises a `TypeError`."""
        a = Parameters({}, 1)

        with pytest.raises(TypeError):
            _ = a < 1
        with pytest.raises(TypeError):
            _ = a > 1
        with pytest.raises(TypeError):
            _ = a <= 1
        with pytest.raises(TypeError):
            _ = a >= 1

    def test_parameter_not_equal_to_numeric(self):
        """Test that `Parameters` instances do not compare equal with numerics of the same value."""
        a = Parameters({}, 1)
        assert a != 1
        # Ignore the "unnecessary negation" error - it's to check we don't have `a == 1 and a != 1`!
        assert not a == 1  # pylint: disable=unnecessary-negation

    def test_clone_state_dict(self):
        """Test that `clone_state_dict` creates a copy of the given tensors."""
        tensor = Tensor([1, 2, 3])
        original_tensor = tensor.clone()
        state_dict = {"testing": tensor}
        # pylint: disable-next=protected-access
        state_dict_cloned = Parameters._clone_state_dict(state_dict)
        state_dict_assigned = state_dict

        # Make a change to the original tensor
        tensor[0] = 2

        # The state dict that was copied just by assignment reflects the change...
        torch.testing.assert_close(state_dict_assigned["testing"], tensor)

        # ...but the cloned state dict doesn't
        torch.testing.assert_close(state_dict_cloned["testing"], original_tensor)

    def test_clone_state_dict_usage(self):
        """Test that the `Parameters` class makes immutable clones of the state dictionaries passed to it."""
        tensor = Tensor([1, 2, 3])
        original_tensor = tensor.clone()
        state_dict = {"testing": tensor}
        module = Mock(spec=Module)
        parameters = Parameters({module: state_dict}, 1)

        # Make a change to the module's state dictionary
        state_dict["testing"][0] = 2

        # The Parameters object should not reflect this change
        torch.testing.assert_close(parameters.parameters[module]["testing"], original_tensor)
