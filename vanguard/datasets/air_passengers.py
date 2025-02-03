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

from importlib.resources import as_file, files

import numpy as np
import pandas as pd

from vanguard.datasets.basedataset import Dataset


class AirPassengers(Dataset):
    """
    Analysis of air passengers through time.

    Functionality to load the air passengers dataset, taken from :cite:`Jiang_KATS_2022`.
    """

    def __init__(self) -> None:
        """Initialise self."""
        super().__init__(np.array([]), 0.0, np.array([]), 0.0, np.array([]), 0.0, np.array([]), 0.0, 0.0)

    @staticmethod
    def _load_data() -> pd.DataFrame:
        """
        Load the data.

        :return: A data frame containing the air passengers data.
        """
        file_name = "air_passengers.csv"
        try:
            with as_file(files("vanguard.datasets.data").joinpath(file_name)) as f:
                df = pd.read_csv(f)
        except FileNotFoundError as exc:
            message = f"Could not find data at {file_name}."
            raise FileNotFoundError(message) from exc
        return df
