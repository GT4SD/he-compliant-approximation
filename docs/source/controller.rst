Model Approximation Controller
==============================

.. autoclass:: src.hela.approximation.controller.ModelApproximationController
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

The ``ModelApproximationController`` is responsible for managing the approximation of neural network modules. It is initialized with a model and a set of modules to be approximated. It provides methods to update the model, the set of modules to approximate, and the saving path. It also includes functionality to print the model structure, the approximated model structure, and the available approximators.

Usage
-----

Below is an example of how to use the ``ModelApproximationController`` to approximate a model:

.. code-block:: python

   from hela.approximation.controller import ModelApproximationController
   from hela.approximation.module_to_approximate import ToApproximate
   from your_model_definition import YourModel

   # Instantiate your model
   model = YourModel()

   # Define the modules to approximate
   to_approximate = ToApproximate(modules_set={...})

   # Initialize the approximation controller
   controller = ModelApproximationController(model=model, to_approximate=to_approximate)

   # Get the approximated model
   approximated_model = controller.get_approximated_model(pretrained=False)

This example demonstrates initializing the ``ModelApproximationController`` with a model and a set of modules to approximate, and then retrieving the approximated model.
