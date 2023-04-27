"""Module approximator for attention masking."""

from typing import Any, Dict, List, Tuple

from torch import Tensor, nn

from ...core import ModuleApproximator
from ...multihead.customizable_multihead import _attn_masking


class MultiplicativeAttentionMaskingApproximator(ModuleApproximator):
    """Handles the approximation of the masking process in a multihead attention module.

    Attributes:
        supported_layer_types: contains the classes of the modules or functions that the approximator can approximate.
        approximation_type: name to identify the approximator referring to the type of approximation module.
        is_approximation_trainable: establishes if the approximation contain some trainable parameters.
    """

    supported_layer_types = {_attn_masking}
    approximation_type = "multiplicative"
    is_approximation_trainable = False

    def __init__(
        self, parameters: Dict[str, Any] = {}, **kwargs: Dict[str, Any]
    ) -> None:
        """Initializes the MultiplicativeAttentionMaskingApproximator.

        Args:
            parameters: parameters of the MultiplicativeAttentionMasking modules. Defaults to {}.
        """
        super().__init__(parameters, **kwargs)
        self.approximations: List[MultiplicativeAttentionMasking] = []

    def get_trainable_approximation(self, **kwargs: Dict[str, Any]) -> nn.Module:
        """Approximates the module for the training phase.

        Returns:
            approximated module ready for the training phase.
        """
        new_approximation = MultiplicativeAttentionMasking(**self.parameters)
        # adding the module to the approximation list
        self.approximations.append(new_approximation)
        return new_approximation

    def get_pretrained_approximation(
        self, module: nn.Module, **kwargs: Dict[str, Any]
    ) -> nn.Module:
        """Converts the trainable approximation of the module into its pretrained form.

        Args:
            module: module approximation to be converted.

        Raises:
            ValueError: this method must be called for MultiplicativeAttentionMasking modules.

        Returns:
            approximated module in its pretrained form.
        """
        if not isinstance(module, MultiplicativeAttentionMasking):
            raise ValueError(
                f"{module.__class__} is not a {MultiplicativeAttentionMasking}"
            )
        return module


class MultiplicativeAttentionMasking(nn.Module):
    """Attention masking through mask multiplication.

    Attributes:
        is_approximation_of: class of the approximated module/function.
    """

    is_approximation_of = _attn_masking

    def __init__(self, attn_mask_value: float = 0.0) -> None:
        """Initializes the MultiplicativeAttentionMasking.

        Args:
            attn_mask_value: masking value (i.e. what normally is -inf). Defaults to 0.0 .
        """
        super().__init__()
        self.attn_mask_value = attn_mask_value

    def _fill_mask(self, mask: Tensor) -> Tensor:
        """Creation of a multiplicative attention mask

        Args:
            mask: attention mask to be updated.

        Returns:
            updated mask compatible with multiplicative masking.
        """
        # since masking is applied thorugh multiplication to retain the values we multiply by 1
        mask[mask == 0] = 1
        # updating the masking values
        mask[mask == float("-inf")] = self.attn_mask_value
        return mask

    def forward(
        self, attn: Tensor, attn_mask: Tensor, *args: Tuple, **kwargs: Dict[str, Any]
    ) -> Tensor:
        """Attention masking through mask multiplication

        Args:
            attn: attention values.
            attn_mask: attention mask.

        Returns:
            masked attention values.
        """
        if attn_mask is not None:
            attn = attn * self._fill_mask(attn_mask)
        return attn
