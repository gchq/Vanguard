"""
Enable Bayesian hyperparameters in a controller.

Applying the :class:`~vanguard.hierarchical.module.BayesianHyperparameters`
to a module will make its parameters Bayesian, so posteriors can be inferred
over them rather than point estimates.
"""

from functools import partial
from typing import Any, Iterable, List, Optional, Tuple, TypeVar

import gpytorch
import torch
from gpytorch import constraints

from vanguard.hierarchical.hyperparameter import BayesianHyperparameter

HALF_INTERVAL_PRIOR = (8.0, 6.0**2)
DEFAULT_PRIORS = {
    constraints.Positive: HALF_INTERVAL_PRIOR,
    constraints.GreaterThan: HALF_INTERVAL_PRIOR,
    constraints.LessThan: HALF_INTERVAL_PRIOR,
    constraints.Interval: (0.0, 1.6**2),
    type(None): (0.0, 10.0**2),
}

ModuleT = TypeVar("ModuleT", bound=gpytorch.module.Module)


class BayesianHyperparameters:
    """
    A decorator to apply to modules (means and kernels).

    It converts the module's point-estimate hyperparameters into Bayesian hyperparameters
    over which inference can be performed.

    .. note::
        Modules decorated with this decorator must be used in a controller decorated with
        a subclass of :class:`~vanguard.hierarchical.base.BaseHierarchicalHyperparameters`.
        Using in a controller without this decorator or in raw GPyTorch will lead to errors.

    .. note::
        This decorator will automatically descend into sub-modules of the class to which it
        is applied only if those module are not passed as arguments to the ``__init__``.
        That is, only parameters that are directly created by the init of the class will be
        affected (even if they are buried in further sub-modules).
    """

    def __init__(
        self,
        ignored_parameters: Optional[Iterable[str]] = frozenset(),
        prior_means: Optional[dict] = None,
        prior_variances: Optional[dict] = None,
    ) -> None:
        """
        Initialise self.

        :param ignored_parameters: Names of module hyperparameters which should not be converted
            to Bayesian parameters.
        :param prior_means: Dict of mean values for the prior distributions
        :param prior_variances: Dict of variances for the prior distributions
        """
        self.ignored_parameters = set(ignored_parameters)

        # Work out each ignored parameter to add, then update the set at once, as looping over it
        # whilst also changing its size won't work
        ignored_parameters_to_add = []
        for param in self.ignored_parameters:
            ignored_parameters_to_add.append(f"raw_{param}")
        self.ignored_parameters.update(ignored_parameters_to_add)
        self.prior_means = prior_means if prior_means is not None else {}
        self.prior_variances = prior_variances if prior_variances is not None else {}
        self.prior_means.update({f"raw_{param}": value for param, value in self.prior_means.items()})
        self.prior_variances.update({f"raw_{param}": value for param, value in self.prior_variances.items()})

    def __call__(self, module_class: ModuleT) -> ModuleT:
        ignored_parameters = self.ignored_parameters

        process_hyperparameter = partial(
            _process_hyperparameter, prior_means=self.prior_means, prior_variances=self.prior_variances
        )

        class InnerClass(module_class):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.bayesian_hyperparameters = []

                top_level_parameter_names, lower_level_parameter_names = _discover_modules_and_parameters(self, args)

                for raw_parameter_name in top_level_parameter_names:
                    if raw_parameter_name not in ignored_parameters:
                        self.bayesian_hyperparameters.append(process_hyperparameter(self, raw_parameter_name))

                for module, raw_parameter_name in lower_level_parameter_names:
                    if not hasattr(module, "bayesian_hyperparameters"):
                        module.bayesian_hyperparameters = []
                    module.bayesian_hyperparameters.append(process_hyperparameter(module, raw_parameter_name))

        return InnerClass


def _process_hyperparameter(
    module: ModuleT, raw_parameter_name: str, prior_means: Optional[dict], prior_variances: Optional[dict]
) -> BayesianHyperparameter:
    """
    Determine the information to construct a torch parameter.

    Extract necessary information about a module's parameter, then remove the parameter
    and return the information. The information returned is sufficient to construct
    a hierarchical Bayesian variational version of the same parameter.

    :param module: The module that owns the parameter.
    :param raw_parameter_name: The name of parameter as an attribute of the module.

    :returns: BayesianHyperparameter object with attributes:
        * parameter raw_name [str]
        * parameter raw_shape [torch.Size]
        * parameter constraint [gpytorch.constraints.Constraint,None]
        * parameter prior_mean [float] - the mean of the diagonal normal prior on the raw parameter
        * parameter prior_variance [float] - the variance of the diagonal normal prior on the raw parameter
    """
    constraint = getattr(module, f"{raw_parameter_name}_constraint", None)
    prior_mean = prior_means.get(raw_parameter_name, None)
    prior_variance = prior_variances.get(raw_parameter_name, None)
    raw_parameter_shape = getattr(module, raw_parameter_name).shape
    if prior_mean is None or prior_variance is None:
        prior_mean, prior_variance = DEFAULT_PRIORS[type(constraint)]
    return BayesianHyperparameter(raw_parameter_name, raw_parameter_shape, constraint, prior_mean, prior_variance)


def _discover_modules_and_parameters(
    top_module: torch.nn.Module, base_modules: Iterable[torch.nn.Module]
) -> Tuple[List[str], List[Tuple[ModuleT, str]]]:
    """
    Recursively identify all recursive modules and corresponding parameters.

    Identify all parameters of the supplied module and all parameters of some sub-modules
    along with the sub-module themselves. Excluded are any sub-modules that appear in
    the supplied collection of base modules. For example, if top_module were a ScaledKernel
    applied to an RBFKernel, and base_modules contained that RBFKernel, then only the parameters
    of the ScaleKernel would be identified.

    :param top_module: The module whose parameters will be discovered.
    :param base_modules: The modules that will be ignored if found
                                                    to be sub-modules of top_module.

    :returns:
        * The parameters belonging directly to the top_module.
        * The parameters belonging to sub-modules of the top_module.
    """
    top_level_parameter_names, lower_level_parameter_names = [], []
    for parameter_name, _ in top_module.named_parameters():
        ancestry = parameter_name.split(".")
        if len(ancestry) == 1:
            top_level_parameter_names.append(ancestry[0])
        else:
            sub_module = getattr(top_module, ancestry[0])
            if sub_module not in base_modules:
                lower_level_parameter_names.append(_descend_module_tree(sub_module, ancestry[1:]))
    return top_level_parameter_names, lower_level_parameter_names


def _descend_module_tree(top_module: ModuleT, parameter_ancestry: List[str]) -> Tuple[ModuleT, str]:
    """
    Step through the submodules of a GPyTorch module to find the sub module in direct possession of a parameter.

    :param top_module: The top module to search through.
    :param parameter_ancestry: Descending list of submodule names and the param name.

    :returns:
        * The module in direct possession of the parameter.
        * The parameter name only as an attribute of the returned module.
    """
    if len(parameter_ancestry) == 1:
        return top_module, parameter_ancestry[0]
    else:
        return _descend_module_tree(getattr(top_module, parameter_ancestry[0]), parameter_ancestry[1:])
