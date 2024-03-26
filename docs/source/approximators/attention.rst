Attention
=========

This section documents the approximators designed for attention mechanisms within the `attention` subfolder. These approximators showcase the diverse approaches to approximating key components of attention mechanisms, catering to different requirements in terms of approximation strategies and flexibility.

Multiplicative Attention Masking Approximator
---------------------------------------------

.. automodule:: src.hela.approximation.approximators.attention.masking.multiplicative
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :noindex:

The ``MultiplicativeAttentionMaskingApproximator`` handles the approximation of the masking process in a multihead attention module. It provides a mechanism for approximating attention masking through mask multiplication.

Not Scaled Query-Key Dot Product Approximator
---------------------------------------------

.. automodule:: src.hela.approximation.approximators.attention.query_key_product.not_scaled
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :noindex:

The ``NotScaledQueryKeyDotProductApproximator`` focuses on approximating the query-key product in a multihead attention module without scaling. It offers a fixed approximation for the dot product computation between query and key matrices.