"""Pytest fixtures shared across all tests."""

import numpy as np
import pytest
import torch.random


@pytest.fixture(autouse=True)
def _set_random_seeds():
    """Set the global random seed for numpy and pytorch to ensure reproducibility of tests."""
    torch.random.manual_seed(0x1234_5678_9ABC_DEF)

    # While usage of np.random.* directly is deprecated, and shouldn't happen anywhere within the codebase,
    # we still set the random seed here to ensure reproducibility if it is accidentally used.
    np.random.seed(0x1234_5678)  # noqa: NPY002
