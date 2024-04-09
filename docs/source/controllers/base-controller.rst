Base Controller Class
=====================

A controller class is designed to emulate the boiler plate code which often needs to be written in order to use mod:`gpytorch`.

GP Controller
-------------

All controllers should inherit from the following class:

.. autoclass:: vanguard.base.gpcontroller.GPController
    :show-inheritance:
    :members:
    :member-order: bysource


Posteriors
----------

.. automodule:: vanguard.base.posteriors

.. autoclass:: vanguard.base.posteriors.Posterior
    :members:
    :member-order: bysource

.. autoclass:: vanguard.base.posteriors.MonteCarloPosteriorCollection
    :members:
    :member-order: bysource
    :private-members: _decide_mc_num_samples


Base Controller
---------------

.. automodule:: vanguard.base.basecontroller
    :members:
    :member-order: bysource
    :private-members:
    :exclude-members: posterior_class, posterior_collection_class


Metrics
-------

.. automodule:: vanguard.base.metrics
    :members:
    :exclude-members: MetricsTracker

.. autoclass:: vanguard.base.metrics.MetricsTracker
    :members:
