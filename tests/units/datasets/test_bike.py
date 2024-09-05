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

"""Tests for `vanguard.datasets.bike`."""

import warnings
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tests.cases import get_default_rng
from vanguard.datasets.bike import BikeDataset


class TestBikeDataset(TestCase):
    """Tests for the :class:`vanguard.datasets.bike.BikeDataset` class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up data shared across tests."""
        # Mock data to avoid reading from disk on every unittest run
        cls.mocked_data = pd.DataFrame(
            [
                {
                    "dteday": pd.to_datetime("2024-03-04"),
                    "instant": 3.0,
                    "casual": 4.0,
                    "registered": 5.0,
                    "var_1": 10.0,
                },
                {
                    "dteday": pd.to_datetime("2024-03-05"),
                    "instant": 4.0,
                    "casual": 3.0,
                    "registered": 4.0,
                    "var_1": 20.0,
                },
                {
                    "dteday": pd.to_datetime("2024-03-06"),
                    "instant": 5.0,
                    "casual": 2.0,
                    "registered": 3.0,
                    "var_1": 30.0,
                },
            ]
        )

        # Define inputs to the dataset and create the dataset using the mocked data
        cls.num_samples = 3
        cls.training_proportion = 2.0 / 3.0
        cls.significance = 0.015
        cls.noise_scale = 0.0005
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = cls.mocked_data.copy()
            cls.dataset = BikeDataset(
                cls.num_samples, cls.training_proportion, cls.significance, cls.noise_scale, get_default_rng()
            )

    def test_num_points(self) -> None:
        """Test that the dataset is generated with the correct number of points."""
        assert self.dataset.num_training_points == int(self.num_samples * self.training_proportion)
        assert self.dataset.num_testing_points == int(self.num_samples * (1 - self.training_proportion))
        assert self.dataset.num_points == self.num_samples

    @pytest.mark.no_beartype
    def test_get_n_samples(self) -> None:
        """Test the various possible outcomes of getting a number of samples."""
        data = np.array([1.0, 2.0, 3.0])

        # pylint: disable=protected-access
        # If we request a valid number of samples, we should just get out what we put in
        assert self.dataset._get_n_samples(data, n_samples=1) == 1

        # If we request too many samples, we should just use all data points but get a warning that this
        # is being done
        with warnings.catch_warnings(record=True) as warn:
            assert self.dataset._get_n_samples(data, n_samples=10) == self.mocked_data.shape[0]
        assert len(warn) == 1
        assert (
            str(warn[0].message)
            == "You requested more samples than there are data points in the data. Using all data points instead."
        )

        # If we request a negative number of samples, we should not be able to proceed
        with self.assertRaisesRegex(ValueError, "A negative number of samples has been requested."):
            self.dataset._get_n_samples(data, n_samples=-2)

        # If we don't specify how many samples we want, we should use them all
        assert self.dataset._get_n_samples(data, n_samples=None) == self.mocked_data.shape[0]

        # If we request a non-integer number of samples, we should not be able to proceed
        with self.assertRaisesRegex(ValueError, "A non-integer number of samples has been requested."):
            self.dataset._get_n_samples(data, n_samples=2.0)

        # pylint: enable=protected-access

    def test_data_loading(self) -> None:
        """Test loading a file and processing it."""
        # The expected output comes from converting the column 'dteday' in self.mocked_data to an integer based on
        # the day, and retaining only the values in column 'var_1' from the remaining columns
        expected_output = np.array([[4.0, 10.0], [5.0, 20.0], [6.0, 30.0]])
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = self.mocked_data
            # pylint: disable-next=protected-access
            np.testing.assert_array_equal(self.dataset._load_data(), expected_output)

    def test_data_loading_file_not_found(self) -> None:
        """Test loading a file when it cannot be found on disk."""

        def forced_error(file_path: str, parse_dates: list):
            """Force a ``FileNotFoundError`` to be returned regardless of input parameters."""
            raise FileNotFoundError("Test")

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.side_effect = forced_error
            with self.assertRaisesRegex(FileNotFoundError, "Could not find data"):
                # pylint: disable-next=protected-access
                self.dataset._load_data()
