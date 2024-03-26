Activation Function
===================

This section documents the approximators designed for activation functions within the `activation` subfolder. These approximators are specialized for different types of activation functions, demonstrating the flexibility of the approximation framework, providing both fixed and trainable approximations.

Quadratic Approximator
----------------------

.. automodule:: src.hela.approximation.approximators.activation.quadratic
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :noindex:

The ``QuadraticApproximator`` is designed to approximate ReLU activation functions using a quadratic function. It is a non-trainable approximator that provides a simple, fixed approximation.

Trainable Quadratic Approximator
--------------------------------

.. automodule:: src.hela.approximation.approximators.activation.trainable_quadratic
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :noindex:

The ``TrainableQuadraticApproximator`` targets ReLU activation functions as well but introduces trainable parameters to allow for a more flexible approximation. It includes a mechanism for a smooth transition from the ReLU function to the (trained) quadratic approximation.

