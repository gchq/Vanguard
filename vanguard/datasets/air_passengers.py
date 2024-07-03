"""
The air passengers dataset contains information air travel through time.

This dataset is taken from the Kats Repository in the Facebook research repo, see :cite:`Jiang_KATS_2022`.
"""

import numpy as np
import pandas as pd

from .basedataset import FileDataset


class AirPassengers(FileDataset):
    """
    Analysis of air passengers through time.

    Functionality to load the air passengers dataset, taken from :cite:`Jiang_KATS_2022`.
    """

    def __init__(self):
        """
        Initialise self.

        We do not need any functionality from the FileDataset class so we instead just use null values
        to initialise. The real value of subclassing here is loading and downloading data in the unified
        interface.
        """
        super().__init__(np.array([]), 0.0, np.array([]), 0.0, np.array([]), 0.0, np.array([]), 0.0, 0.0)

    def _load_data(self) -> pd.DataFrame:
        """Load the data."""
        file_path = self._get_data_path("air_passengers.csv")
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError as exc:
            message = (
                f"Could not find data at {file_path}. If you have not downloaded the data, "
                f"call {type(self).__name__}.download()."
            )
            raise FileNotFoundError(message) from exc
        return df

    @classmethod
    def download(cls) -> None:
        """Download the dataset."""
        raise NotImplementedError(
            "Dataset download not implemented for air passengers data. This must instead be done manually "
            "from the Github repo."
        )
