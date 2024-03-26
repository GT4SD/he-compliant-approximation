Softmax Approximators
=====================

This section documents the approximators designed for softmax functions within the `softmax` subfolder. These approximators focus on approximating softmax operations, offering various alternatives, such as polynomial and Taylor expansion, for the approximation of the softmax function.

MLPSoftmax Approximator
-----------------------

.. automodule:: src.hela.approximation.approximators.softmax.mlp_softmax
   :members:
   :show-inheritance:
   :special-members: __init__
   :noindex:

The ``LitMLPSoftmax`` class represents a PyTorch Lightning model for MLPSoftmax. It provides functionalities for training and validation steps specific to the MLPSoftmax model.

PolynomialSoftmax Approximator
-------------------------------

.. automodule:: src.hela.approximation.approximators.softmax.polynomial
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :noindex:

The ``PolynomialSoftmaxApproximator`` handles the approximation of the softmax function using polynomial approximations. It offers both trainable and fixed approximations for the softmax function, allowing for customization of the polynomial order and normalization options.

TaylorSoftmax Approximator
--------------------------

.. automodule:: src.hela.approximation.approximators.softmax.taylor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :noindex:

The ``TaylorSoftmaxApproximator`` focuses on approximating the softmax function using Taylor series expansions. It provides a non-trainable approximation based on the specified order of the Taylor series, offering an alternative approach to approximating softmax operations.