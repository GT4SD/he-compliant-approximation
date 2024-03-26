Pooling Approximators
=====================

This section documents the approximators designed for pooling layers within the `pooling` subfolder. These approximators focus on approximating pooling mechanisms, offering specialized approximations for different types of pooling operations.

AvgPooling2d Approximator
-------------------------

.. automodule:: src.hela.approximation.approximators.pooling.avg_pooling_2d
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :noindex:

The ``AvgPooling2dApproximator`` handles the approximation of max pooling over 2 dimensions. It provides a non-trainable approximation for `nn.MaxPool2d` layers, offering a simple and fixed approximation for pooling operations through average pooling.
