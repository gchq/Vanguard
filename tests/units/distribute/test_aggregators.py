"""
Tests for aggregator classes.
"""
import unittest

import torch

from vanguard.distribute import aggregators


class AggregationTests(unittest.TestCase):
    """
    Test that the results are the same.

    Note that these tests are for reproducibility, and assume that the results
    of the functions were originally correct.
    """
    def setUp(self) -> None:
        """Code to run before each test."""
        self.means = [
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1, 3, 1, 1]),
            torch.tensor([-1, -4, 5, 8])
        ]

        self.covars = [
            torch.tensor([[0.3, 0, 0, 0],
                          [0, 0.3, 0, 0],
                          [0, 0, 0.3, 0],
                          [0, 0, 0, 0.3]]),
            torch.tensor([[0.1, 0, 0, 0],
                          [0, 0.2, 0, 0],
                          [0, 0, 1.0, 0],
                          [0, 0, 0, 1.3]]),
            torch.tensor([[0.4, 0, 0, 0],
                          [0, 0.5, 0, 0],
                          [0, 0, 0.7, 0],
                          [0, 0, 0, 0.2]])
        ]

        self.prior_var = torch.tensor([0.2, 0.2, 0.2, 0.2])

        self.expected_means_and_variances = {
            aggregators.POEAggregator: (torch.tensor([0.68421053, 1.32258065, 3.14876033, 5.94366197]),
                                        torch.tensor([0.06315789, 0.09677419, 0.17355372, 0.10985915])),
            aggregators.EKPOEAggregator: (torch.tensor([0.68421053, 1.32258065, 3.14876033, 5.94366197]),
                                          torch.tensor([0.18947368, 0.29032258, 0.52066116, 0.32957746])),
            aggregators.GPOEAggregator: (torch.tensor([0.68421053, 1.32258065, 3.14876033, 5.94366197]),
                                         torch.tensor([0.18947368, 0.29032258, 0.52066116, 0.32957746])),
            aggregators.BCMAggregator: (torch.tensor([1.857143, 41.00004, -4.280899, -60.285694]),
                                        torch.tensor([0.171429, 3.000003, -0.235955, -1.114285])),
            aggregators.RBCMAggregator: (torch.tensor([0.46066617, 0.34468132, -0.67688588, -0.36816736]),
                                         torch.tensor([0.12598918, 0.1489797,  0.09264543, 0.10755615])),
            aggregators.XBCMAggregator: (torch.tensor([0.83592976, 1.32969308, 0.20348966, 2.67023639]),
                                         torch.tensor([0.13819973, 0.21805526, 0.1729148, 0.16951545])),
            aggregators.GRBCMAggregator: (torch.tensor([1.07106864, 3.51013531, 1.22329068, 5.57793316]),
                                          torch.tensor([0.09881552, 0.18724662, 0.55341864, 0.90324579])),
            aggregators.XGRBCMAggregator: (torch.tensor([0.87888387, 2.63632948, 0.94631851, 8.55604603]),
                                           torch.tensor([0.1020186, 0.20909176, 1.10736298, 0.64514268]))
        }

    def test_output_types(self) -> None:
        """Should all be tensors."""
        for aggregator_class in self.expected_means_and_variances:
            with self.subTest(aggregator_class=aggregator_class.__name__):
                aggregator = aggregator_class(self.means, self.covars, self.prior_var)
                observed_mean, observed_variance = aggregator.aggregate()
                self.assertIsInstance(observed_mean, torch.Tensor)
                self.assertIsInstance(observed_variance, torch.Tensor)

    def test_output_values(self) -> None:
        """Should all be correct."""
        for aggregator_class, (mean, variance) in self.expected_means_and_variances.items():
            with self.subTest(aggregator_class=aggregator_class.__name__):
                aggregator = aggregator_class(self.means, self.covars, self.prior_var)
                observed_mean, observed_variance = aggregator.aggregate()
                torch.testing.assert_allclose(mean, observed_mean)
                torch.testing.assert_allclose(variance, observed_variance.diagonal())
