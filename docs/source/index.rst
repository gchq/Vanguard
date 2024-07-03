.. figure:: _static/logo.png

Welcome to Vanguard's Documentation!
====================================

Vanguard is a high-level wrapper around `GPyTorch <https://gpytorch.ai>`_ and aims to provide a user-friendly interface
for training and using Gaussian process models. Vanguard's main objective is to make a variety of more advanced GP
techniques in the machine learning literature available for easy use by a non-specialists and specialists alike.
Vanguard is designed for modularity to facilitate straightforward combinations of different techniques.

Vanguard was created by GCHQ.

.. toctree::
    :maxdepth: 1
    :caption: Examples

    examples


.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    examples/decorator_walkthrough.ipynb


.. toctree::
    :maxdepth: 1
    :caption: Components

    components/base-gp-models
    components/kernels


.. toctree::
    :maxdepth: 1
    :caption: Controllers

    controllers/base-controller
    controllers/vanilla
    controllers/input-uncertainty


.. toctree::
    :maxdepth: 1
    :caption: Decorators

    decorators/warped-gps
    decorators/learning-likelihood-noise
    decorators/normalising-inputs
    decorators/variational-inference
    decorators/distributed-gps
    decorators/classification
    decorators/multitask
    decorators/decorator-tools
    decorators/input-warping.rst
    decorators/hierarchical.rst
    decorators/higher-rank-features
    decorators/disable-standard-scaling


.. toctree::
    :maxdepth: 1
    :caption: Optimisation

    optimise/smart_optimiser
    optimise/finder
    optimise/schedulers

.. toctree::
    :maxdepth: 1
    :caption: Datasets

    datasets/base-datasets
    datasets/bike
    datasets/air_passengers
    datasets/synthetic


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References
==========

.. bibliography::
