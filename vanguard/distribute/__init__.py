"""
Distribute training and prediction across multiple controllers.

A controller class decorated with the  class:`~vanguard.distribute.decorator.Distributed` decorator
will make predictions by aggregating the predictions of several independent expert controllers.
"""
from .decorator import Distributed
