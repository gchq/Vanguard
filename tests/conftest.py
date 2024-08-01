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

"""Pytest fixtures shared across all tests."""

import os
import random

import numpy as np
import pytest
import torch.random


@pytest.fixture(autouse=True)
def _set_random_seeds():
    """Set the global random seed for various random generators to ensure reproducibility of tests."""
    torch.random.manual_seed(0x1234_5678_9ABC_DEF)

    # While usage of np.random.* directly is deprecated, and shouldn't happen anywhere within the codebase,
    # we still set the random seed here to ensure reproducibility if it is accidentally used.
    np.random.seed(0x1234_5678)  # noqa: NPY002

    # Additional seeds that we set just in case
    random.seed(0x1234_5678)
    os.environ["PYTHONHASHSEED"] = str(0x1234_5678)
