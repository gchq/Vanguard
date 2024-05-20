from unittest import TestCase

import numpy as np

from vanguard.classification.likelihoods import DirichletKernelClassifierLikelihood


class TestDirichletKernelClassifierLikelihood(TestCase):
    def test_illegal_input_type(self):
        """Test that we get an appropriate error when an illegal argument type is passed."""
        likelihood = DirichletKernelClassifierLikelihood(num_classes=10)

        # various illegal inputs
        illegal_inputs = [object(), np.array([1, 2, 3]), "string"]

        for illegal_input in illegal_inputs:
            with self.subTest(repr(illegal_input)):
                with self.assertRaises(RuntimeError) as ctx:
                    # ignore type: it's intentionally incorrect
                    likelihood(illegal_input)  # type: ignore
                self.assertEqual(
                    "Likelihoods expects a DummyKernelDistribution input to make marginal predictions, or a "
                    f"torch.Tensor for conditional predictions. Got a {type(illegal_input).__name__}",
                    ctx.exception.args[0],
                )
