Classification
==============

.. automodule:: vanguard.classification

.. automodule:: vanguard.classification.mixin
    :members:

Binary Classification
---------------------

.. autoclass:: vanguard.classification.binary.BinaryClassification
    :show-inheritance:
    :members:

Categorical Classification
--------------------------

.. autoclass:: vanguard.classification.categorical.CategoricalClassification
    :show-inheritance:
    :members:

.. autoclass:: vanguard.classification.likelihoods.MultitaskBernoulliLikelihood
    :members:

.. autoclass:: vanguard.classification.likelihoods.SoftmaxLikelihood
    :members:


Multiclass Classification
-------------------------

.. autoclass:: vanguard.classification.dirichlet.DirichletMulticlassClassification
    :show-inheritance:
    :members:

.. autoclass:: vanguard.classification.kernel.DirichletKernelMulticlassClassification
    :show-inheritance:
    :members:

.. autoclass:: vanguard.classification.likelihoods.DirichletKernelClassifierLikelihood
    :members:
    :exclude-members: forward

.. autoclass:: vanguard.classification.likelihoods.DirichletKernelDistribution
    :members:

.. autoclass:: vanguard.classification.likelihoods.GenericExactMarginalLogLikelihood
    :members:

.. autoclass:: vanguard.classification.models.InertKernelModel
    :members:
