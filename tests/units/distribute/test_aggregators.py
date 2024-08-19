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
Tests for aggregator classes.
"""

import unittest

import torch

from vanguard.distribute import aggregators


class AggregationTests(unittest.TestCase):
    """
    Test that the results from aggregation are as expected.

    Note that these tests are for reproducibility, and assume that the results
    of the functions were originally correct.
    """

    def setUp(self) -> None:
        """Define data shared across tests."""
        self.means = [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 3, 1, 1]), torch.tensor([-1, -4, 5, 8])]

        self.covars = [
            torch.tensor(
                [
                    [0.3, 0, 0, 0],
                    [0, 0.3, 0, 0],
                    [0, 0, 0.3, 0],
                    [0, 0, 0, 0.3],
                ]
            ),
            torch.tensor(
                [
                    [0.1, 0, 0, 0],
                    [0, 0.2, 0, 0],
                    [0, 0, 1.0, 0],
                    [0, 0, 0, 1.3],
                ]
            ),
            torch.tensor(
                [
                    [0.4, 0, 0, 0],
                    [0, 0.5, 0, 0],
                    [0, 0, 0.7, 0],
                    [0, 0, 0, 0.2],
                ]
            ),
        ]

        self.prior_var = torch.tensor([0.2, 0.2, 0.2, 0.2])

        self.expected_means_and_variances = {
            aggregators.POEAggregator: (
                torch.tensor([0.68421053, 1.32258065, 3.14876033, 5.94366197]),
                torch.tensor([0.06315789, 0.09677419, 0.17355372, 0.10985915]),
            ),
            aggregators.EKPOEAggregator: (
                torch.tensor([0.68421053, 1.32258065, 3.14876033, 5.94366197]),
                torch.tensor([0.18947368, 0.29032258, 0.52066116, 0.32957746]),
            ),
            aggregators.GPOEAggregator: (
                torch.tensor([0.68421053, 1.32258065, 3.14876033, 5.94366197]),
                torch.tensor([0.18947368, 0.29032258, 0.52066116, 0.32957746]),
            ),
            aggregators.BCMAggregator: (
                torch.tensor([1.857143, 41.00004, -4.280899, -60.285694]),
                torch.tensor([0.171429, 3.000003, -0.235955, -1.114285]),
            ),
            aggregators.RBCMAggregator: (
                torch.tensor([0.46066617, 0.34468132, -0.67688588, -0.36816736]),
                torch.tensor([0.12598918, 0.1489797, 0.09264543, 0.10755615]),
            ),
            aggregators.XBCMAggregator: (
                torch.tensor([0.83592976, 1.32969308, 0.20348966, 2.67023639]),
                torch.tensor([0.13819973, 0.21805526, 0.1729148, 0.16951545]),
            ),
            aggregators.GRBCMAggregator: (
                torch.tensor([1.07106864, 3.51013531, 1.22329068, 5.57793316]),
                torch.tensor([0.09881552, 0.18724662, 0.55341864, 0.90324579]),
            ),
            aggregators.XGRBCMAggregator: (
                torch.tensor([0.87888387, 2.63632948, 0.94631851, 8.55604603]),
                torch.tensor([0.1020186, 0.20909176, 1.10736298, 0.64514268]),
            ),
        }

    def test_output_types(self) -> None:
        """
        Test that the outputs from aggregation are the expected tensors.

        When predictions from multiple experts are combined, the resulting combinations
        should be tensors, regardless of the aggregation method used.
        """
        for aggregator_class in self.expected_means_and_variances:
            with self.subTest(aggregator_class=aggregator_class.__name__):
                aggregator = aggregator_class(self.means, self.covars, self.prior_var)
                observed_mean, observed_variance = aggregator.aggregate()
                self.assertIsInstance(observed_mean, torch.Tensor)
                self.assertIsInstance(observed_variance, torch.Tensor)

    def test_output_values(self) -> None:
        """
        Test that the outputs from aggregation match expected values.

        The expected means and variances after aggregation are written in the setup of
        this class. Here we test that these are indeed produced when the inputs are aggregated.
        """
        for aggregator_class, (mean, variance) in self.expected_means_and_variances.items():
            with self.subTest(aggregator_class=aggregator_class.__name__):
                aggregator = aggregator_class(self.means, self.covars, self.prior_var)
                observed_mean, observed_variance = aggregator.aggregate()
                torch.testing.assert_close(mean, observed_mean)
                torch.testing.assert_close(variance, observed_variance.diagonal())

    def test_invalid_prior_var(self) -> None:
        """
        Test that the BaseAggregator rejects an invalid `prior_var` input.

        If the shape of `prior_var` does not match the shape of the provided variances
        (that is, the diagonal of the provided covariances), there must have been an issue
        with the input data, so the code should identify this and raise an appropriate error.
        We check that both an error is raised, and the text matches our expectation.
        """
        with self.assertRaises(ValueError) as exc:
            aggregators.BaseAggregator(self.means, self.covars, torch.tensor([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(
            str(exc.exception), "Prior var shape torch.Size([2, 3]) doesn't match variances shape torch.Size([3, 4])"
        )

    def test_invalid_aggregation_with_base_class(self) -> None:
        """
        Test that if one tries to perform aggregation with the base class, it does not work.

        Aggregation methods are to be specified for each child class of BaseAggregator, so if
        we try to aggregate with the base class, we should not get any result.
        """
        with self.assertRaises(NotImplementedError):
            aggregator = aggregators.BaseAggregator(self.means, self.covars, self.prior_var)
            aggregator.aggregate()
