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
The user-facing interface of the :class:`~vanguard.base.basecontroller.BaseGPController` class.
"""

import warnings
from typing import List, Optional, Type, Union

import numpy as np
import numpy.typing
import torch
from torch import Tensor
from typing_extensions import Self

from vanguard.base.basecontroller import BaseGPController
from vanguard.base.metaclass import _StoreInitValues
from vanguard.base.metrics import MetricsTracker
from vanguard.base.posteriors.posterior import Posterior
from vanguard.decoratorutils import Decorator


class GPController(BaseGPController, metaclass=_StoreInitValues):
    """
    The base class for GP controllers.

    The following class variables will persist unless changed (manually or by decorators):

    * ``device``: The device of the tensors to be used. By default, this will be set to ``torch.device('cuda:0')`` if
      :func:`torch.cuda.is_available()` returns ``True``, but otherwise it defaults to ``torch.device('cpu')``.
    * ``dtype``: The dtype for tensors, which defaults to ``torch.float32``. Setting this higher will improve accuracy
      at the cost of memory.
    * ``gp_model_class``: An uninstantiated subclass of :class:`~gpytorch.models.ExactGP` or
      :class:`~gpytorch.models.ApproximateGP` to be used in inference.
    * ``posterior_class``: An uninstantiated subclass of :class:`~vanguard.base.posteriors.Posterior` to be used for
      all posteriors returned during prediction.
    * ``posterior_collection_class``: An uninstantiated subclass of
      :class:`~vanguard.base.posteriors.MonteCarloPosteriorCollection` to be used for all posteriors returned during
      fuzzy prediction.

    .. note::
        The loss after each iteration of hyperparameter tuning is saved in the controller's
        metrics tracker (accessed using the :meth:`metrics_tracker` property), and can be printed
        during fitting by using the :meth:`~vanguard.base.metrics.MetricsTracker.print_metrics` method.
        Consider this example (:class:`~vanguard.vanilla.GaussianGPController` is used
        to simplify the example; metric printing is available for all controller classes):

        :Example:
            >>> from vanguard.datasets.synthetic import SyntheticDataset
            >>> from vanguard.kernels import ScaledRBFKernel
            >>> from vanguard.vanilla import GaussianGPController
            >>>
            >>> dataset = SyntheticDataset()
            >>>
            >>> controller = GaussianGPController(dataset.train_x, dataset.train_y,
            ...                                   ScaledRBFKernel, dataset.train_y_std)
            >>> initial_loss = controller.fit(10)
            >>> with controller.metrics_tracker.print_metrics():
            ...     loss = controller.fit(5)  # doctest: +ELLIPSIS
            iteration: 11, loss: ...
            iteration: 12, loss: ...
            iteration: 13, loss: ...
            iteration: 14, loss: ...
            iteration: 15, loss: ...

        For more options see the :class:`~vanguard.base.metrics.MetricsTracker` class.
    """

    _init_params = {}
    __decorators__: List[Type[Decorator]] = []

    @property
    def likelihood_noise(self) -> Tensor:
        """Return the noise of the likelihood."""
        return self._likelihood.noise

    @likelihood_noise.setter
    def likelihood_noise(
        self,
        value: Tensor,
    ) -> None:
        """Set the noise of the likelihood."""
        self._likelihood.noise = value

    @property
    def learning_rate(self) -> np.floating:
        """Return the learning rate of the parameter optimiser."""
        return self._smart_optimiser.learning_rate

    @learning_rate.setter
    def learning_rate(
        self,
        value: np.floating,
    ) -> None:
        """Set the learning rate of the parameter optimiser."""
        self._smart_optimiser.learning_rate = value

    @property
    def metrics_tracker(self) -> MetricsTracker:
        """Return the :class:`~vanguard.base.metrics.MetricsTracker` associated with the controller."""
        return self._metrics_tracker

    def fit(
        self,
        n_sgd_iters: int = 10,
        gradient_every: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Run rounds of hyperparameter tuning.

        .. note::
            By default ``fit(n_sgd_iters=n, gradient_every=m)`` is equivalent to ``fit(n_sgd_iters=n)``.
            However, any changes to :meth:`~vanguard.base.basecontroller.BaseGPController._sgd_round`
            could break this equivalence.

        .. warning::
            Do **not** overload this method in order to alter SGD behaviour. Instead, overload
            :meth:`~vanguard.base.basecontroller.BaseGPController._sgd_round` to ensure that
            all added functionality propagates correctly.

        :param n_sgd_iters: The number of gradient updates to perform in each round of hyperparameter tuning.
        :param gradient_every: How often (in iterations) to do special HNIGP input gradient steps.
                                    Defaults to same as `n_sgd_iters` normally, overridden to 1 in batch-mode.
        :returns: The loss.
        """
        if self.batch_size is not None:
            if gradient_every is not None:
                warnings.warn(
                    f"You are trying to set gradient_every (in this case to {gradient_every}) in batch mode."
                    "This does not make mathematical sense and your value of gradient every will be ignored "
                    " and replaced by 1."
                )
            gradient_every = 1

        gradient_every = n_sgd_iters if gradient_every is None else gradient_every

        loss = self._sgd_round(n_iters=n_sgd_iters, gradient_every=gradient_every)
        return loss

    def posterior_over_point(
        self,
        x: Union[numpy.typing.NDArray[np.floating], float],
    ) -> Posterior:
        """
        Return predictive posterior of the y-value over a point.

        :param x: (n_predictions, n_features) The predictive inputs.
        :returns: The posterior.
        """
        return self._get_posterior_over_point_in_eval_mode(x)

    def posterior_over_fuzzy_point(
        self,
        x: Union[numpy.typing.NDArray[np.floating], float],
        x_std: Union[numpy.typing.NDArray[np.floating], float],
    ) -> Posterior:
        """
        Return predictive posterior of the y-value over a fuzzy point.

        .. warning:
            The ``n_features`` must match with :attr:`self.dim`.

        :param x: (n_predictions, n_features) The predictive inputs.
        :param x_std: The input noise standard deviations:

            * array_like[np.floating]: (n_features,) The standard deviation per input dimension for the predictions,
            * np.floating: Assume homoskedastic noise.

        :returns: The posterior.
        """
        return self._get_posterior_over_fuzzy_point_in_eval_mode(x, x_std)

    def predictive_likelihood(
        self,
        x: Union[numpy.typing.NDArray[np.floating], float],
    ) -> Posterior:
        """
        Calculate the predictive likelihood at an x-value.

        :param x: (n_predictions, n_features) The points at which to obtain the likelihood.
        :returns: The marginal distribution.
        """
        return self._predictive_likelihood(x)

    def fuzzy_predictive_likelihood(
        self,
        x: Union[numpy.typing.NDArray[np.floating], float],
        x_std: Union[numpy.typing.NDArray[np.floating], float],
    ) -> Posterior:
        """
        Calculate the predictive likelihood at an x-value, given variance.

        :param x: (n_predictions, n_features) The points at which to obtain the likelihood.
        :param x_std: (n_predictions, n_features) The std-dev of input points.
        :returns: The marginal distribution.
        """
        return self._fuzzy_predictive_likelihood(x, x_std)

    @classmethod
    def new(cls, instance: Self, **kwargs) -> Self:
        """
        Create an instance of the class with the same initialisation parameters as an existing instance.

        Any keyword arguments passed in this method will overwrite the values used for the initialisation
        of the new instance. Calling ``type(instance).new(instance)`` is essentially equivalent to creating
        a copy of instance, albeit with some parameters potentially remaining connected.

        .. warning::
            This method is **not** guaranteed to return a deep copy of an instance if the classes match.
            Attributes such as the training data, and kernel are likely to be shared across instances. To
            mitigate this, explicitly pass copies of these as keyword parameters.
        """
        initialisation_params = instance._init_params.copy()  # pylint: disable=protected-access
        initialisation_params.update(kwargs)
        return cls(**initialisation_params)
