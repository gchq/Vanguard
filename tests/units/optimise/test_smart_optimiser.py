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

"""Tests for the SmartOptimiser class."""

import unittest
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import Linear

from vanguard.optimise import NoImprovementError, SmartOptimiser


class TestSmartOptimiser(unittest.TestCase):
    """Tests for the `SmartOptimiser` class."""

    def test_early_stopping(self):
        """Test that the `SmartOptimiser` stops early if there is no improvement across several steps."""
        patience = 5
        smart_optimiser = SmartOptimiser(torch.optim.Adam, Linear(2, 2), early_stop_patience=patience)
        with patch.object(smart_optimiser, "_step", return_value=1.0):
            for _ in range(patience):
                smart_optimiser.step(1.0)
            with pytest.raises(
                NoImprovementError, match=f"Stopping early due to no improvement on {patience} consecutive steps"
            ):
                smart_optimiser.step(1.0)

    def test_no_early_stopping(self):
        """Test that the `SmartOptimiser` doesn't stop early if patience is unset."""
        smart_optimiser = SmartOptimiser(torch.optim.Adam, Linear(2, 2), early_stop_patience=None)
        with patch.object(smart_optimiser, "_step", return_value=1.0):
            # Assertion: we don't hit any NoImprovementError
            for _ in range(100):
                smart_optimiser.step(1.0)

    def test_reset_resets_last_n_losses(self):
        """Test that calling `reset()` resets the last N recorded losses."""
        smart_optimiser = SmartOptimiser(torch.optim.Adam, Linear(2, 2))

        # Take a step, so there's a non-NaN loss in the last N losses
        smart_optimiser.step(1.0)
        assert not all(np.isnan(x) for x in smart_optimiser.last_n_losses)

        # Reset and check that it's removed
        smart_optimiser.reset()
        assert all(np.isnan(x) for x in smart_optimiser.last_n_losses)

    def test_reset_resets_all_modules(self):
        """Test that calling `reset()` resets all modules under optimisation."""
        module = Linear(2, 2)
        initial_weights = module.weight.data.clone()
        smart_optimiser = SmartOptimiser(torch.optim.Adam, module)

        # Make a change to the module parameters
        module.weight.data *= 2

        # Reset, and check we also reset the module
        smart_optimiser.reset()
        torch.testing.assert_allclose(module.weight.data, initial_weights)

    def test_register_module(self):
        """Test the ability to register additional modules after the optimiser is created."""
        module1 = torch.nn.Linear(2, 2)
        module2 = torch.nn.Linear(2, 3)
        smart_optimiser = SmartOptimiser(torch.optim.Adam, module1)

        # Optimiser knows about module1 but not module2
        assert module1 in smart_optimiser._stored_initial_state_dicts  # pylint: disable=protected-access
        assert module2 not in smart_optimiser._stored_initial_state_dicts  # pylint: disable=protected-access

        smart_optimiser.register_module(module2)

        # Optimiser knows about both module1 and module2
        assert module1 in smart_optimiser._stored_initial_state_dicts  # pylint: disable=protected-access
        assert module2 in smart_optimiser._stored_initial_state_dicts  # pylint: disable=protected-access

    def test_update_registered_module(self):
        """
        Test the ability to update registered modules in the optimiser.

        This is really meant to be used when additional parameters are added to a module, but here we test the case
        where we're just changing the module's initial values.
        """
        module = torch.nn.Linear(2, 2)
        initial_weights = module.weight.data.clone()
        smart_optimiser = SmartOptimiser(torch.optim.Adam, module)

        # Make a change to the module parameters
        module.weight.data *= 2

        # Check it's not reflected in the optimiser's cache
        # pylint: disable-next=protected-access
        torch.testing.assert_close(smart_optimiser._stored_initial_state_dicts[module]["weight"].data, initial_weights)

        # Update
        smart_optimiser.update_registered_module(module)

        # Check the new weights are reflected in the optimiser's cache
        torch.testing.assert_close(
            # pylint: disable-next=protected-access
            smart_optimiser._stored_initial_state_dicts[module]["weight"].data,
            module.weight.data,
        )

    def test_update_registered_module_keyerror(self):
        """Test that `update_registered_module` raises a `KeyError` if an unrecognised module is passed."""
        smart_optimiser = SmartOptimiser(torch.optim.Adam, Linear(2, 2))

        with pytest.raises(KeyError):
            smart_optimiser.update_registered_module(Linear(3, 3))
