Approximation Pipeline
======================

This module implements the approximation pipeline, which is responsible for managing and executing the steps involved in approximating a neural network model. The pipeline includes functionalities for:

* training
* validating
* testing 

the model across different approximation steps, **handling checkpoints**, and **managing callbacks** such as early stopping conditions.

.. autoclass:: src.hela.approximation.pipeline.approximation_pipeline.ApproximationPipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__


Usage
-----

Provide examples or additional information on how to use the ``ApproximationPipeline`` class and any other important functionalities within this module.

.. code-block:: python

   from src.hela.approximation.pipeline.approximation_pipeline import ApproximationPipeline
   # Example instantiation and usage of the ApproximationPipeline class
