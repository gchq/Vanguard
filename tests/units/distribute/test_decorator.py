"""
Tests for the Distributed decorator.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from tests.cases import get_default_rng
from vanguard.datasets.synthetic import HeteroskedasticSyntheticDataset, SyntheticDataset
from vanguard.distribute import Distributed, aggregators
from vanguard.distribute.decorator import _create_subset
from vanguard.distribute.partitioners import KMedoidsPartitioner
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.warps import SetWarp, warpfunctions


@Distributed(
    n_experts=10, aggregator_class=aggregators.GRBCMAggregator, ignore_methods=("__init__",), rng=get_default_rng()
)
class DistributedGaussianGPController(GaussianGPController):
    """Test class."""


@Distributed(
    n_experts=10, aggregator_class=aggregators.BCMAggregator, ignore_methods=("__init__",), rng=get_default_rng()
)
class DistributedGaussianGPControllerBCMAggregator(GaussianGPController):
    """Test class."""


@Distributed(
    n_experts=10,
    aggregator_class=aggregators.GRBCMAggregator,
    partitioner_class=KMedoidsPartitioner,
    ignore_methods=("__init__",),
    rng=get_default_rng(),
)
class DistributedGaussianGPControllerKMedoids(GaussianGPController):
    """Test class."""


@Distributed(n_experts=10, aggregator_class=aggregators.GRBCMAggregator, ignore_all=True, rng=get_default_rng())
@SetWarp(warpfunctions.AffineWarpFunction(a=3, b=-1) @ warpfunctions.BoxCoxWarpFunction(0.2), ignore_all=True)
class DistributedWarpedGaussianGPController(GaussianGPController):
    """Test class."""


class InitialisationTests(unittest.TestCase):
    """
    Tests for the initialisation of the decorator.
    """

    def setUp(self) -> None:
        self.rng = get_default_rng()

    def test_cannot_pass_array_as_y_std(self) -> None:
        """
        Test that if `train_y_std` is provided as an array, this input is rejected.

        When a controller has been distributed, noise on the outputs can only take the
        form of an int or float. Here we check that a TypeError is raised if the noise
        is given as an array.
        """
        dataset = HeteroskedasticSyntheticDataset(rng=self.rng)

        if isinstance(dataset.train_y_std, (float, int)):
            self.skipTest(f"The standard deviation should be an array, not '{type(dataset.train_y_std).__name__}'.")

        with self.assertRaises(TypeError):
            DistributedGaussianGPController(
                dataset.train_x, dataset.train_y, ScaledRBFKernel, dataset.train_y_std, rng=self.rng
            )

    def test_no_kernel_with_k_medoids(self) -> None:
        """
        Test incorrect initialisation of the distributed decorator when using the KMedoidsPartitioner.

        The KMedoidsPartitioner requires passing a kernel to the object, so we expect initialisation
        to fail if this is not provided.
        """
        # Define the data - for the purposes of this test we do not need to know the y_std values
        dataset = HeteroskedasticSyntheticDataset(rng=self.rng)

        # Create the class without specifying a kernel; we expect an error telling us we need to specify one
        with self.assertRaisesRegex(TypeError, "missing 1 required positional argument: 'kernel'"):
            DistributedGaussianGPControllerKMedoids(
                dataset.train_x, dataset.train_y, ScaledRBFKernel, 0.01, rng=self.rng
            )

    def test_uninitialised_kernel_with_k_medoids(self) -> None:
        """
        Test incorrect initialisation of the distributed decorator when using the KMedoidsPartitioner.

        One issue that users might run into is erroneously passing a kernel _class_ (e.g. `ScaledRBFKernel`) to the
        initialiser, rather than a kernel _instance_ (e.g. `ScaledRBFKernel()`). We specifically check for this type
        of error, and raise a helpful error early.
        """
        dataset = HeteroskedasticSyntheticDataset(rng=self.rng)

        with self.assertRaisesRegex(TypeError, "Invalid kernel type"):
            DistributedGaussianGPControllerKMedoids(
                dataset.train_x,
                dataset.train_y,
                ScaledRBFKernel,
                0.01,
                partitioner_kwargs={"kernel": ScaledRBFKernel},
                rng=self.rng,
            )

    def test_correct_initialisation_with_k_medoids(self) -> None:
        """Test correct initialisation of the distributed decorator when using the KMedoidsPartitioner."""
        dataset = HeteroskedasticSyntheticDataset(rng=self.rng)

        # Create the class whilst specifying a kernel - this should create without error
        DistributedGaussianGPControllerKMedoids(
            dataset.train_x,
            dataset.train_y,
            ScaledRBFKernel,
            0.01,
            partitioner_kwargs={"kernel": ScaledRBFKernel()},
            rng=self.rng,
        )


class SharedDataTests(unittest.TestCase):
    """
    Tests for the sharing of data among experts.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Define data shared across tests."""
        rng = get_default_rng()
        dataset = SyntheticDataset(rng=rng)

        cls.controller = DistributedGaussianGPController(
            dataset.train_x, dataset.train_y, ScaledRBFKernel, dataset.train_y_std, rng=rng
        )
        cls.warp_controller = DistributedWarpedGaussianGPController(
            dataset.train_x, dataset.train_y, ScaledRBFKernel, dataset.train_y_std, rng=rng
        )
        cls.controller.fit(1)
        cls.warp_controller.fit(1)

    def test_experts_have_been_created(self) -> None:
        """
        Test that, after the controllers have been created and fit, they have at-least 1 expert within them.

        The lists of expert controllers should not be empty, as we have created the controllers and fit the GPs.
        """
        # pylint: disable=protected-access
        self.assertNotEqual(0, len(self.controller._expert_controllers))
        self.assertNotEqual(0, len(self.warp_controller._expert_controllers))

    def test_expert_losses(self) -> None:
        """
        Test that the experts are in fact different through inspection of their losses.

        Each expert should get different data to train on, so when we inspect the losses
        of each expert, they should be different.
        """
        # pylint: disable=protected-access
        unique_expert_losses = set(self.controller.expert_losses())
        self.assertEqual(len(unique_expert_losses), len(self.controller._expert_controllers))

    def test_expert_kernels_are_different(self) -> None:
        """
        Test that the experts have different kernels.

        The set of ids needs to be the correct length.
        """
        # pylint: disable=protected-access
        kernel_ids = {id(expert.kernel) for expert in self.controller._expert_controllers}
        self.assertEqual(len(self.controller._expert_controllers), len(kernel_ids))
        self.assertNotIn(id(self.controller.kernel), kernel_ids)

    def test_expert_means_are_different(self) -> None:
        """
        Test that the experts have different means.

        Each expert should get different data to train on, so when we inspect the means
        of each expert, they should be different.
        """
        # pylint: disable=protected-access
        mean_ids = {id(expert.mean) for expert in self.controller._expert_controllers}
        self.assertEqual(len(self.controller._expert_controllers), len(mean_ids))
        self.assertNotIn(id(self.controller.mean), mean_ids)

    def test_expert_warps_are_identical(self) -> None:
        """
        Test that the same warping is applied with each expert.

        Each expert should get different data to train on, but the data should be warped
        in the same way for each. We verify this by ensuring the set only has one element.
        """
        # pylint: disable=protected-access
        warp_ids = {id(expert.warp) for expert in self.warp_controller._expert_controllers}
        warp_ids.add(id(self.warp_controller.warp))
        self.assertEqual(1, len(warp_ids))

    def test_bad_prior_var_shape(self) -> None:
        """
        Test that GP controllers reject invalid prior variance computations when aggregating expert results.

        We verify that, if a kernel is used that does not give a sensible prior variance value, the GP controller
        will reject this rather than return something that the user may not be aware is unreasonable.
        """
        # Define a new controller and fit it
        gp = DistributedGaussianGPControllerBCMAggregator(
            np.arange(20).reshape(-1, 1),
            2.5 + 0.5 * np.arange(20),
            ScaledRBFKernel,
            0.0,
            rng=get_default_rng(),
        )
        gp.fit(1)

        # Change the kernel on the controller to ensure we hit the case where the posterior prediction
        # computation does not make sense due to the prior variance computed
        mocked_kernel = MagicMock()
        mocked_kernel.return_value = torch.zeros(size=[2, 3, 5])
        gp.kernel = mocked_kernel

        # Check we reject the invalid noise from the kernel
        with self.assertRaises(RuntimeError) as exc:
            gp.posterior_over_point(np.arange(3))
        self.assertEqual(
            str(exc.exception), "Cannot distribute using this kernel - try using a non-BCM aggregator instead."
        )


class SubsetCreationTests(unittest.TestCase):
    """
    Tests for the creation of data subsets used by each expert.
    """

    def test_create_subset_expected_inputs(self):
        """
        Test the handling of expected inputs (arrays with the shape attribute) when calling _create_subset.

        We pass two numpy arrays to _create_subset, and expect the result from _create_subset to return subsets
        of each.
        """
        # Define arrays to subset
        first_array = np.array([1, 2, 3, 4])
        second_array = np.array([5, 6, 7, 8])

        # Subset the arrays - setting subset_fraction such that we expect 2 points to be
        # taken from each array passed
        subset_arrays = _create_subset(first_array, second_array, subset_fraction=0.5, rng=get_default_rng())

        # Regardless of random seed, package version and so on, we expect the results to have the following properties:
        # Each of the two resulting subset arrays has exactly 2 elements, both taken from the corresponding input array
        # with no duplicates
        self.assertEqual(len(subset_arrays), 2)
        self.assertEqual(len(subset_arrays[0]), 2)
        self.assertEqual(len(subset_arrays[1]), 2)
        self.assertTrue(set(subset_arrays[0]) <= set(first_array))
        self.assertTrue(set(subset_arrays[1]) <= set(second_array))
        self.assertTrue(len(set(subset_arrays[0])) == 2)
        self.assertTrue(len(set(subset_arrays[1])) == 2)

    def test_create_subset_unexpected_inputs(self):
        """
        Test the handling of arrays without the shape attribute.

        When using the function _create_subset and passing arrays that do not have the
        shape attribute, we should simply get the input returned as a list.
        """
        with self.assertWarns(Warning) as warning_raised:
            self.assertListEqual(
                [[[1, 2, 3, 4], [5, 6, 7, 8]]],
                _create_subset([[1, 2, 3, 4], [5, 6, 7, 8]], subset_fraction=0.5, rng=get_default_rng()),
            )
        # To ensure an unrelated warning is raised, check the warning text is as expected
        self.assertEqual(
            str(warning_raised.warning),
            "Input 'arrays' are expected to be numpy arrays or floats. Got an array of type `list' which will not be "
            "split into a subset.",
        )
