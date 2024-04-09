"""
Tests for the Distributed decorator.
"""
import unittest

from vanguard.datasets.synthetic import (HeteroskedasticSyntheticDataset,
                                         SyntheticDataset)
from vanguard.distribute import Distributed, aggregators
from vanguard.kernels import ScaledRBFKernel
from vanguard.vanilla import GaussianGPController
from vanguard.warps import SetWarp, warpfunctions


@Distributed(n_experts=10, aggregator_class=aggregators.GRBCMAggregator, ignore_methods=("__init__",))
class DistributedGaussianGPController(GaussianGPController):
    """Test class."""
    pass


@Distributed(n_experts=10, aggregator_class=aggregators.GRBCMAggregator, ignore_all=True)
@SetWarp(warpfunctions.AffineWarpFunction(a=3, b=-1) @ warpfunctions.BoxCoxWarpFunction(0.2), ignore_all=True)
class DistributedWarpedGaussianGPController(GaussianGPController):
    """Test class."""
    pass


class InitialisationTests(unittest.TestCase):
    """
    Tests for the initialisation of the decorator.
    """
    def test_cannot_pass_array_as_y_std(self):
        """Should raise a TypeError."""
        dataset = HeteroskedasticSyntheticDataset()

        if isinstance(dataset.train_y_std, (float, int)):
            self.skipTest(f"The standard deviation should be an array, not '{type(dataset.train_y_std).__name__}'.")

        with self.assertRaises(TypeError):
            DistributedGaussianGPController(dataset.train_x, dataset.train_y,
                                            ScaledRBFKernel, dataset.train_y_std)


class SharedDataTests(unittest.TestCase):
    """
    Testing the sharing of data among experts.
    """
    @classmethod
    def setUpClass(cls):
        """Code to run before all tests."""
        dataset = SyntheticDataset()

        cls.controller = DistributedGaussianGPController(dataset.train_x, dataset.train_y, ScaledRBFKernel,
                                                         dataset.train_y_std)
        cls.warp_controller = DistributedWarpedGaussianGPController(dataset.train_x, dataset.train_y, ScaledRBFKernel,
                                                                    dataset.train_y_std)
        cls.controller.fit(1)
        cls.warp_controller.fit(1)

    def test_experts_have_been_created(self):
        """Lists should not be empty."""
        self.assertNotEqual(0, len(self.controller._expert_controllers))
        self.assertNotEqual(0, len(self.warp_controller._expert_controllers))

    def test_expert_losses(self):
        """All losses should be different."""
        unique_expert_losses = set(self.controller.expert_losses())
        self.assertEqual(len(unique_expert_losses), len(self.controller._expert_controllers))

    def test_expert_kernels_are_different(self):
        """The set of ids needs to be the correct length."""
        kernel_ids = {id(expert.kernel) for expert in self.controller._expert_controllers}
        self.assertEqual(len(self.controller._expert_controllers), len(kernel_ids))
        self.assertNotIn(id(self.controller.kernel), kernel_ids)

    def test_expert_means_are_different(self):
        """The set of ids needs to be the correct length."""
        mean_ids = {id(expert.mean) for expert in self.controller._expert_controllers}
        self.assertEqual(len(self.controller._expert_controllers), len(mean_ids))
        self.assertNotIn(id(self.controller.mean), mean_ids)

    def test_expert_warps_are_identical(self):
        """The set should only have one element."""
        warp_ids = {id(expert.warp) for expert in self.warp_controller._expert_controllers}
        warp_ids.add(id(self.warp_controller.warp))
        self.assertEqual(1, len(warp_ids))
