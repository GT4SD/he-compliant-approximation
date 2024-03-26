Module Approximators
====================

.. automodule:: src.hela.approximation.approximators.core
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :noindex:

The ``ModuleApproximator`` abstract class serves as the foundation for creating specific approximators for different types of neural network modules. It defines a common interface and essential functionalities that all approximators must implement outlining the structure and the required methods for approximating neural network modules. It includes methods for approximating a module, configuring optimizers, and handling training steps. Child classes must implement these methods to provide specific approximation logic.

Creating Child Approximators
----------------------------

To create a specific approximator, inherit from the ``ModuleApproximator`` class and implement the abstract methods. Each child approximator should specify the types of layers it supports, the approximation type, and whether the approximation contains trainable parameters.

Example:

.. code-block:: python

   class CustomApproximator(ModuleApproximator):
       supported_layer_types = {nn.Linear}
       approximation_type = "custom"
       is_approximation_trainable = True

       def approximate_module(self, model, id, pretrained, **kwargs):
           # Implementation for approximating a module
           pass

       # Implement other abstract methods...

This example demonstrates how to create a custom approximator for `nn.Linear` layers by inheriting from `ModuleApproximator` and implementing the required methods.

Child approximators are located in the subfolders of `approximators`, each tailored to approximate specific types of neural network modules, such as activation functions, attention mechanisms, or layer normalizations.

.. toctree::
   :hidden:

   activation
   attention
   layer_normalization
   multihead
   pooling
   softmax