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
The air passengers dataset contains information air travel through time.

This dataset is taken from the Kats Repository in the Facebook research repo, see :cite:`Jiang_KATS_2022`.
"""

import numpy as np
import pandas as pd

from vanguard.datasets.basedataset import FileDataset


class AirPassengers(FileDataset):
    """
    Analysis of air passengers through time.

    Functionality to load the air passengers dataset, taken from :cite:`Jiang_KATS_2022`.

    We do not need any functionality from the :class:`~vanguard.datasets.FileDataset` class so we instead just use null
    values to initialise. The real value of subclassing here is loading and downloading data in the unified interface.
    """

    def __init__(self) -> None:
        """Initialise self."""
        super().__init__(np.array([]), 0.0, np.array([]), 0.0, np.array([]), 0.0, np.array([]), 0.0, 0.0)

    def _load_data(self) -> pd.DataFrame:
        """
        Load the data.

        :return: A data frame containing the air passengers data.
        """
        file_path = self._get_data_path("air_passengers.csv")
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError as exc:
            if __debug__:
                message = f"Could not find data at {file_path}."
            else:
                message = "Could not find data at `vanguard/datasets/data/air_passengers.csv`."
            raise FileNotFoundError(message) from exc
        return df
