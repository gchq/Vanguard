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

"""Tests for :mod:`vanguard.datasets.bike`."""

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tests.cases import get_default_rng
from vanguard.datasets.bike import BikeDataset


class TestBikeDataset:
    """Tests for the :class:`vanguard.datasets.bike.BikeDataset` class."""

    # Define inputs to the dataset
    num_samples = 3
    training_proportion = 0.8
    significance = 0.015
    noise_scale = 0.0005

    @pytest.fixture
    def mocked_data(self) -> pd.DataFrame:
        """Set up mock data to avoid reading from disk on every unittest run."""
        return pd.DataFrame(
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

    @pytest.fixture
    def dataset(self, mocked_data: pd.DataFrame) -> BikeDataset:
        """Set up a BikeDataset instance without reading from disk."""
        # Define inputs to the dataset and create the dataset using the mocked data
        with patch("pandas.read_csv", return_value=mocked_data.copy()):
            return BikeDataset(
                self.num_samples, self.training_proportion, self.significance, self.noise_scale, rng=get_default_rng()
            )

    def test_num_points(self, dataset: BikeDataset) -> None:
        """Test that the dataset is generated with the correct number of points."""
        assert dataset.num_training_points == math.ceil(self.num_samples * self.training_proportion)
        assert dataset.num_testing_points == math.floor(self.num_samples * (1 - self.training_proportion))
        assert dataset.num_points == self.num_samples

    @pytest.mark.parametrize("training_proportion", [-1.0, 2.0])
    def test_invalid_training_proportion(self, mocked_data: pd.DataFrame, training_proportion: float) -> None:
        """Test that setting a training proportion outside the range (0, 1) raises an appropriate error."""
        with patch("pandas.read_csv", return_value=mocked_data):
            with pytest.raises(ValueError, match="`training_proportion` must be between 0 and 1"):
                BikeDataset(training_proportion=training_proportion, rng=get_default_rng())

    def test_data_loading(self, dataset: BikeDataset, mocked_data: pd.DataFrame) -> None:
        """Test loading a file and processing it."""
        # The expected output comes from converting the column 'dteday' in self.mocked_data to an integer based on
        # the day, and retaining only the values in column 'var_1' from the remaining columns
        expected_output = np.array([[4.0, 10.0], [5.0, 20.0], [6.0, 30.0]])
        with patch("pandas.read_csv", return_value=mocked_data):
            # pylint: disable-next=protected-access
            np.testing.assert_array_equal(dataset._load_data(), expected_output)

    def test_data_loading_file_not_found(self, dataset: BikeDataset) -> None:
        """Test loading a file when it cannot be found on disk."""

        def forced_error(file_path: str, parse_dates: list):
            """Force a ``FileNotFoundError`` to be returned regardless of input parameters."""
            raise FileNotFoundError("Test")

        with patch("pandas.read_csv", side_effect=forced_error):
            with pytest.raises(FileNotFoundError, match="Could not find data"):
                # pylint: disable-next=protected-access
                dataset._load_data()


class TestGetNSamples:
    """Tests for :meth:`vanguard.datasets.bike.BikeDataset._get_n_samples`."""

    data = np.array([1.0, 2.0, 3.0])
    # pylint: disable=protected-access

    def test_valid(self) -> None:
        """If we request a valid number of samples, we should just get out what we put in."""
        assert BikeDataset._get_n_samples(self.data, n_samples=1) == 1

    def test_request_too_many(self) -> None:
        """If we request too many samples, we should just use all data points but get an appropriate warning."""
        with pytest.warns(
            match="You requested more samples than there are data points in the data. Using all data points instead."
        ):
            assert BikeDataset._get_n_samples(self.data, n_samples=10) == self.data.shape[0]

    def test_request_negative(self) -> None:
        """If we request a negative number of samples, we should not be able to proceed."""
        with pytest.raises(ValueError, match="A negative number of samples has been requested."):
            BikeDataset._get_n_samples(self.data, n_samples=-2)

    def test_unspecified(self) -> None:
        """If we don't specify how many samples we want, we should use them all."""
        assert BikeDataset._get_n_samples(self.data, n_samples=None) == self.data.shape[0]

    @pytest.mark.no_beartype
    def test_non_integer(self) -> None:
        """If we request a non-integer number of samples, we should not be able to proceed."""
        with pytest.raises(TypeError, match="A non-integer number of samples has been requested."):
            BikeDataset._get_n_samples(self.data, n_samples=2.0)

    # pylint: enable=protected-access
