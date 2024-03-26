Layer Normalization
===================

This section documents the approximators designed for layer normalization modules within the `layer_normalization` subfolder. These approximators showcase the diverse approaches to approximating layer normalization mechanisms, offering both fixed and trainable approximations, and catering to different requirements in terms of approximation strategies and optimization techniques.

LayerNorm to BatchNorm Approximator
-----------------------------------

.. automodule:: src.hela.approximation.approximators.layer_normalization.batch_normalization
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :noindex:
   

The ``LayerNormToBatchNormApproximator`` handles the approximation of layer normalization with batch normalization. It provides a specialized module, `BatchNorm1dForTransformers`, adapted for transformers' input dimensions, ensuring compatibility with the layer normalization setup.

Distill LayerNorm Approximator
------------------------------

.. automodule:: src.hela.approximation.approximators.layer_normalization.distill_layernorm
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :noindex:

The ``DistillLayerNormApproximator`` focuses on approximating the layer normalization. It includes a custom optimizer configuration to distill the layer normalization module knowledge into its approximation, enabling efficient training and knowledge transfer.