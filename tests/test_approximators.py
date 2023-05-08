"""Testing the modules' approximators."""

from copy import deepcopy
from typing import Tuple

import pytest
import torch
from torch import Tensor, nn

from hela.approximation.approximators.activation.quadratic import (
    QuadraticApproximation,
    QuadraticApproximator,
)
from hela.approximation.approximators.activation.trainable_quadratic import (
    PairedReLU,
    TrainableQuadraticApproximation,
    TrainableQuadraticApproximator,
)
from hela.approximation.approximators.attention.masking.multiplicative import (
    MultiplicativeAttentionMasking,
    MultiplicativeAttentionMaskingApproximator,
)
from hela.approximation.approximators.attention.query_key_product.not_scaled import (
    NotScaledQueryKeyDotProduct,
    NotScaledQueryKeyDotProductApproximator,
)
from hela.approximation.approximators.core import ModuleApproximator
from hela.approximation.approximators.layer_normalization.batch_normalization import (
    BatchNorm1dForTransformers,
    LayerNormToBatchNormApproximator,
)
from hela.approximation.approximators.layer_normalization.distill_layernorm import (
    DistillLayerNormApproximation,
    DistillLayerNormApproximator,
    PairedLayerNorm,
)
from hela.approximation.approximators.multihead.customizable_multihead import (
    CustomizableMultiHead,
    CustomizableMultiHeadApproximator,
)
from hela.approximation.approximators.softmax.mlp_softmax import (
    MLPSoftmaxApproximation,
    MLPSoftmaxApproximator,
)
from hela.approximation.approximators.softmax.polynomial import (
    PolynomialSoftmax,
    PolynomialSoftmaxApproximator,
)
from hela.approximation.approximators.softmax.taylor import (
    TaylorSoftmax,
    TaylorSoftmaxApproximator,
)

# default device to run tests
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining some testing parameters
batch_size = 10
sequence_length = embedding_dim = 256


# each dictionary entry represent an approximator class and its testing values
# for each 'input_parameters' a corresponding 'expected_output' must be provided
testing_informations = {
    "QuadraticApproximator": {
        "approximator_class": QuadraticApproximator,
        "init_parameters": [{}],
        "trainable_approximation_class": QuadraticApproximation,
        "get_trainable_approximation_kwargs": [{}],
        "pretrained_approximation_class": QuadraticApproximation,
        "forward_kwargs": {
            "input": torch.ones(
                (batch_size, sequence_length, sequence_length), device=DEVICE
            )
            * 2
        },
        "output_type": Tensor,
        "expected_output": [
            torch.ones((batch_size, sequence_length, sequence_length), device=DEVICE)
            * 4,
        ],
    },
    "TrainableQuadraticApproximator": {
        "approximator_class": TrainableQuadraticApproximator,
        "init_parameters": [{"input_dimension": sequence_length}],
        "trainable_approximation_class": PairedReLU,
        "get_trainable_approximation_kwargs": [{}],
        "pretrained_approximation_class": TrainableQuadraticApproximation,
        "forward_kwargs": {
            "input": torch.ones(
                (batch_size, sequence_length, sequence_length), device=DEVICE
            )
            * 2
        },
        "output_type": Tensor,
        "expected_output": [
            torch.ones((batch_size, sequence_length, sequence_length), device=DEVICE)
            * 2,
        ],
    },
    "LayerNormToBatchNormApproximator": {
        "approximator_class": LayerNormToBatchNormApproximator,
        "init_parameters": [{"num_features": embedding_dim}],
        "trainable_approximation_class": BatchNorm1dForTransformers,
        "get_trainable_approximation_kwargs": [{}],
        "pretrained_approximation_class": BatchNorm1dForTransformers,
        "forward_kwargs": {
            "input": torch.ones(
                (batch_size, sequence_length, embedding_dim), device=DEVICE
            )
            * 2
        },
        "output_type": Tensor,
        "expected_output": [
            None,
        ],
    },
    "DistillLayernormApproximator": {
        "approximator_class": DistillLayerNormApproximator,
        "init_parameters": [{"normalized_shape": embedding_dim}],
        "trainable_approximation_class": PairedLayerNorm,
        "get_trainable_approximation_kwargs": [
            {"layernorm": nn.LayerNorm(embedding_dim)}
        ],
        "pretrained_approximation_class": DistillLayerNormApproximation,
        "forward_kwargs": {
            "input": torch.ones(
                (batch_size, sequence_length, embedding_dim), device=DEVICE
            )
            * 2
        },
        "output_type": Tensor,
        "expected_output": [
            None,
        ],
    },
    "CustomizableMultiHeadApproximator": {
        "approximator_class": CustomizableMultiHeadApproximator,
        "init_parameters": [
            {},
            {
                "embed_dim": embedding_dim,
                "num_heads": int(embedding_dim / 2),
                "dropout": 0.5,
            },
        ],
        "trainable_approximation_class": CustomizableMultiHead,
        "get_trainable_approximation_kwargs": [
            {
                "multihead": nn.MultiheadAttention(
                    embed_dim=embedding_dim, num_heads=1
                ),
            },
            {
                "multihead": nn.MultiheadAttention(
                    embed_dim=embedding_dim, num_heads=1
                ),
            },
        ],
        "pretrained_approximation_class": CustomizableMultiHead,
        "forward_kwargs": {
            "query": torch.ones((batch_size, embedding_dim), device=DEVICE),
            "key": torch.ones((batch_size, embedding_dim), device=DEVICE),
            "value": torch.ones((batch_size, embedding_dim), device=DEVICE),
        },
        "output_type": Tuple,
        "expected_output": [
            None,
            None,
        ],
    },
    "PolynomialSoftmaxApproximator": {
        "approximator_class": PolynomialSoftmaxApproximator,
        "init_parameters": [
            {
                "order": 2,
                "dim": -1,
                "skip_normalization": False,
                "init_alpha": 1,
            },
            {
                "order": 2,
                "dim": -1,
                "skip_normalization": True,
                "init_alpha": 1,
            },
            {
                "order": 4,
                "dim": -1,
                "skip_normalization": True,
                "init_alpha": 1,
            },
            {
                "order": 2,
                "dim": -1,
                "skip_normalization": True,
                "init_alpha": 1,
            },
        ],
        "trainable_approximation_class": PolynomialSoftmax,
        "get_trainable_approximation_kwargs": [
            {
                "softmax": nn.Softmax(dim=-1),
            },
            {
                "softmax": nn.Softmax(dim=-1),
            },
            {
                "softmax": nn.Softmax(dim=-1),
            },
            {
                "softmax": PolynomialSoftmax(
                    order=2,
                    dim=-1,
                    skip_normalization=True,
                    init_alpha=1,
                ),
            },
        ],
        "pretrained_approximation_class": PolynomialSoftmax,
        "forward_kwargs": {
            "input": torch.ones(
                (batch_size, sequence_length, sequence_length), device=DEVICE
            )
            * 2,
        },
        "output_type": Tensor,
        "expected_output": [
            torch.ones((batch_size, sequence_length, sequence_length), device=DEVICE)
            / sequence_length,
            torch.pow(
                torch.ones(
                    (batch_size, sequence_length, sequence_length), device=DEVICE
                )
                * 2,
                2,
            ),
            torch.pow(
                torch.ones(
                    (batch_size, sequence_length, sequence_length), device=DEVICE
                )
                * 2,
                4,
            ),
            torch.pow(
                torch.ones(
                    (batch_size, sequence_length, sequence_length), device=DEVICE
                )
                * 2,
                2,
            ),
        ],
    },
    "TaylorSoftmaxApproximator": {
        "approximator_class": TaylorSoftmaxApproximator,
        "init_parameters": [{}],
        "trainable_approximation_class": TaylorSoftmax,
        "get_trainable_approximation_kwargs": [{}],
        "pretrained_approximation_class": TaylorSoftmax,
        "forward_kwargs": {
            "input": torch.ones(
                (batch_size, sequence_length, sequence_length), device=DEVICE
            ),
        },
        "output_type": Tensor,
        "expected_output": [
            torch.ones((batch_size, sequence_length, sequence_length), device=DEVICE)
            / sequence_length,
        ],
    },
    "MLPSoftmaxApproximator": {
        "approximator_class": MLPSoftmaxApproximator,
        "init_parameters": [{"dim_size": embedding_dim, "unit_test": True}],
        "trainable_approximation_class": MLPSoftmaxApproximation,
        "get_trainable_approximation_kwargs": [{}],
        "pretrained_approximation_class": MLPSoftmaxApproximation,
        "forward_kwargs": {
            "input": torch.ones(
                (batch_size, sequence_length, sequence_length), device=DEVICE
            ),
        },
        "output_type": Tensor,
        "expected_output": [
            None,
        ],
    },
    "MultiplicativeAttentionMaskingApproximator": {
        "approximator_class": MultiplicativeAttentionMaskingApproximator,
        "init_parameters": [
            {"attn_mask_value": 0.0},
            {"attn_mask_value": 1.0},
        ],
        "trainable_approximation_class": MultiplicativeAttentionMasking,
        "get_trainable_approximation_kwargs": [{}, {}],
        "pretrained_approximation_class": MultiplicativeAttentionMasking,
        "forward_kwargs": {
            "attn": torch.ones((sequence_length, sequence_length), device=DEVICE),
            "attn_mask": torch.triu(
                torch.ones((sequence_length, sequence_length), device=DEVICE)
                * float("-inf"),
                diagonal=1,
            ),
        },
        "output_type": Tensor,
        "expected_output": [
            torch.tril(
                torch.ones((sequence_length, sequence_length), device=DEVICE),
                diagonal=0,
            ),
            torch.ones((sequence_length, sequence_length), device=DEVICE),
        ],
    },
    "NotScaledQueryKeyProductApproximator": {
        "approximator_class": NotScaledQueryKeyDotProductApproximator,
        "init_parameters": [{}],
        "trainable_approximation_class": NotScaledQueryKeyDotProduct,
        "get_trainable_approximation_kwargs": [{}],
        "pretrained_approximation_class": NotScaledQueryKeyDotProduct,
        "forward_kwargs": {
            "query": torch.ones(
                (batch_size, sequence_length, embedding_dim), device=DEVICE
            ),
            "key": torch.ones(
                (batch_size, sequence_length, embedding_dim), device=DEVICE
            )
            * 2,
        },
        "output_type": Tensor,
        "expected_output": [
            torch.ones((batch_size, sequence_length, sequence_length), device=DEVICE)
            * 2
            * embedding_dim,
        ],
    },
}


@pytest.mark.parametrize(
    "approximator_identifier",
    list(testing_informations.keys()),
    ids=list(testing_informations.keys()),
)
def test_approximator_init(
    approximator_identifier: str,
):
    """Tests the initialization of the approximator.

    Args:
        approximator_identifier: identifier of the approximator to be tested.
    """
    # retrieving the testing values for the approximator
    approximator_dictionary = testing_informations[approximator_identifier]

    for init_parameters in approximator_dictionary["init_parameters"]:
        # initializing the approximator object
        approximator = approximator_dictionary["approximator_class"](
            parameters=init_parameters
        )

        # ASSERTS

        # checking the approximator class
        assert isinstance(approximator, ModuleApproximator)
        assert isinstance(approximator, approximator_dictionary["approximator_class"])


@pytest.mark.parametrize(
    "approximator_identifier",
    list(testing_informations.keys()),
    ids=list(testing_informations.keys()),
)
def test_get_trainable_approximation(
    approximator_identifier: str,
):
    """Tests the approximator trainable approximation.

    Args:
        approximator_identifier: identifier of the approximator to be tested.
    """
    # retrieving the testing values for the approximator
    approximator_dictionary = testing_informations[approximator_identifier]

    for index, init_parameters in enumerate(approximator_dictionary["init_parameters"]):
        # initializing the approximator object
        approximator = approximator_dictionary["approximator_class"](
            parameters=init_parameters
        )
        # getting the trainable approximation
        trainable_approx_module = approximator.get_trainable_approximation(
            **approximator_dictionary["get_trainable_approximation_kwargs"][index]
        )

        # ASSERTS

        # checking the trainable approximation class
        assert isinstance(
            trainable_approx_module,
            approximator_dictionary["trainable_approximation_class"],
        )


@pytest.mark.parametrize(
    "approximator_identifier",
    list(testing_informations.keys()),
    ids=list(testing_informations.keys()),
)
def test_trainable_approximation_forward(
    approximator_identifier: str,
):
    """Tests the forward pass of the approximator trainable approximation.

    Args:
        approximator_identifier: identifier of the approximator to be tested.
    """
    # retrieving the testing values for the approximator
    approximator_dictionary = testing_informations[approximator_identifier]

    for index, init_parameters in enumerate(approximator_dictionary["init_parameters"]):
        # initializing the approximator object
        approximator = approximator_dictionary["approximator_class"](
            parameters=init_parameters
        )
        # getting the trainable approximation
        trainable_approx_module = approximator.get_trainable_approximation(
            **approximator_dictionary["get_trainable_approximation_kwargs"][index]
        )
        # moving the approximated module to the default DEVICE
        trainable_approx_module.to(DEVICE)
        # forward pass through the trainable approximation
        output = trainable_approx_module(
            **deepcopy(approximator_dictionary["forward_kwargs"])
        )

        # ASSERTS

        # checking the output type
        assert isinstance(output, approximator_dictionary["output_type"])
        # checking the output values if they are specified
        if approximator_dictionary["expected_output"][index] is not None:
            assert torch.all(
                torch.eq(
                    output,
                    approximator_dictionary["expected_output"][index],
                )
            )


@pytest.mark.parametrize(
    "approximator_identifier",
    list(testing_informations.keys()),
    ids=list(testing_informations.keys()),
)
def test_get_pretrained_approximation(
    approximator_identifier: str,
):
    """Tests the approximator pretrained approximation.

    Args:
        approximator_identifier: identifier of the approximator to be tested.
    """
    # retrieving the testing values for the approximator
    approximator_dictionary = testing_informations[approximator_identifier]

    for index, init_parameters in enumerate(approximator_dictionary["init_parameters"]):
        # initializing the approximator object
        approximator = approximator_dictionary["approximator_class"](
            parameters=init_parameters
        )
        # getting the trainable approximation
        trainable_approx_module = approximator.get_trainable_approximation(
            **approximator_dictionary["get_trainable_approximation_kwargs"][index]
        )
        # getting the pretrained approximation
        pretrained_approx_module = approximator.get_pretrained_approximation(
            trainable_approx_module
        )

        # ASSERTS

        # checking the trainable approximation class
        assert isinstance(
            trainable_approx_module,
            approximator_dictionary["trainable_approximation_class"],
        )
        # checking the pretrained approximation class
        assert isinstance(
            pretrained_approx_module,
            approximator_dictionary["pretrained_approximation_class"],
        )


@pytest.mark.parametrize(
    "approximator_identifier",
    list(testing_informations.keys()),
    ids=list(testing_informations.keys()),
)
def test_pretrained_approximation_forward(approximator_identifier: str):
    """Tests the forward pass of the approximator pretrained approximation.

    Args:
        approximator_identifier: identifier of the approximator to be tested.
    """
    # retrieving the testing values for the approximator
    approximator_dictionary = testing_informations[approximator_identifier]

    for index, init_parameters in enumerate(approximator_dictionary["init_parameters"]):
        # initializing the approximator object
        approximator = approximator_dictionary["approximator_class"](
            parameters=init_parameters
        )
        # getting the trainable approximation
        trainable_approx_module = approximator.get_trainable_approximation(
            **approximator_dictionary["get_trainable_approximation_kwargs"][index]
        )
        # getting the pretrained approximation
        pretrained_approx_module = approximator.get_pretrained_approximation(
            trainable_approx_module
        )
        # moving the approximated module to the default DEVICE
        pretrained_approx_module.to(DEVICE)
        # forward pass through the pretrained approximation
        output = pretrained_approx_module(
            **deepcopy(approximator_dictionary["forward_kwargs"])
        )

        # ASSERTS

        # checking the output type
        assert isinstance(output, approximator_dictionary["output_type"])
        # checking the output values if they are specified
        if approximator_dictionary["expected_output"][index] is not None:
            assert torch.all(
                torch.eq(
                    output,
                    approximator_dictionary["expected_output"][index],
                )
            )
