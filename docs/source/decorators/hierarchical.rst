Hierarchical GPs with Bayesian Hyperparameters
==============================================

.. automodule:: vanguard.hierarchical


Variational Hierarchical
------------------------

.. automodule:: vanguard.hierarchical.variational
    :show-inheritance:
    :members:

Laplace Hierarchical
--------------------

.. automodule:: vanguard.hierarchical.laplace
    :show-inheritance:
    :members:

Base Hierarchical
--------------------

.. automodule:: vanguard.hierarchical.base
    :show-inheritance:
    :members:


Hierarchical Bayesian Kernels and Means
---------------------------------------

.. automodule:: vanguard.hierarchical.module
    :members:


Bayesian Hyperparameters
------------------------

Hyperparameters are held within the controller inside a class:`~vanguard.hierarchical.collection.HyperparameterCollection` instance,
comprised of instances of the lightweight class:`~vanguard.hierarchical.hyperparameter.BayesianHyperparameter` class.

.. autoclass:: vanguard.hierarchical.collection.HyperparameterCollection
    :members:

.. autoclass:: vanguard.hierarchical.hyperparameter.BayesianHyperparameter
    :members:
