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

"""Tests for `vanguard.datasets.air_passengers`."""

from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from vanguard.datasets.air_passengers import AirPassengers


class TestAirPassengersDataset(TestCase):
    """Tests for the `AirPassengers` class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up data shared across tests."""
        # Mock data to avoid reading from disk on every unittest run
        cls.mocked_data = pd.DataFrame([{"ds": "2024-03-04", "y": 3.0}, {"ds": "2024-03-05", "y": 4.0}])
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = cls.mocked_data.copy()
            cls.dataset = AirPassengers()

    def test_data_loading(self) -> None:
        """Test loading a file and processing it."""
        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = self.mocked_data
            # No processing of any form should be applied, so the data loading method should return the passed
            # mocked data exactly as is
            # pylint: disable-next=protected-access
            pd.testing.assert_frame_equal(self.dataset._load_data(), self.mocked_data)

    def test_data_loading_file_not_found(self) -> None:
        """Test loading a file when it cannot be found on disk."""

        def forced_error(file_path: str):
            """Force a FileNotFoundError to be returned regardless of input parameters."""
            raise FileNotFoundError("Test")

        with patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.side_effect = forced_error
            with self.assertRaisesRegex(FileNotFoundError, "Could not find data"):
                # pylint: disable-next=protected-access
                self.dataset._load_data()
