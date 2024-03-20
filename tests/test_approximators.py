"""Testing the modules' approximators."""

from copy import deepcopy
from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor, nn

from hela.approximation.approximators.activation.quadratic import (
    QuadraticActivation,
    QuadraticApproximator,
)
from hela.approximation.approximators.activation.trainable_quadratic import (
    PairedReLU,
    TrainableQuadraticActivation,
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
    DistillLayerNorm,
    DistillLayerNormApproximator,
    PairedLayerNorm,
)
from hela.approximation.approximators.multihead.customizable_multihead import (
    CustomizableMultiHead,
    CustomizableMultiHeadApproximator,
)
from hela.approximation.approximators.pooling.avg_pooling_2d import (
    AvgPooling2d,
    AvgPooling2dApproximator,
)
from hela.approximation.approximators.softmax.mlp_softmax import (
    MLPSoftmax,
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

# default device to run the tests
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# defining some testing parameters
batch_size = 10
sequence_length = embedding_dim = 256

kernel_size = 32
img_size = 32


# each dictionary entry represent an approximator class and its testing values
# for each 'input_parameters' a corresponding 'expected_output' must be provided
testing_informations = {
    "QuadraticApproximator": {
        "approximator_class": QuadraticApproximator,
        "init_parameters": [{}],
        "trainable_approximation_class": QuadraticActivation,
        "get_trainable_approximation_kwargs": [{}],
        "pretrained_approximation_class": QuadraticActivation,
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
        "pretrained_approximation_class": TrainableQuadraticActivation,
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
        "pretrained_approximation_class": DistillLayerNorm,
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
        "trainable_approximation_class": MLPSoftmax,
        "get_trainable_approximation_kwargs": [{}],
        "pretrained_approximation_class": MLPSoftmax,
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
    "AvgPooling2dApproximator": {
        "approximator_class": AvgPooling2dApproximator,
        "init_parameters": [
            {
                "kernel_size": kernel_size,
                "stride": None,
                "padding": 0,
                "ceil_mode": False,
                "count_include_pad": True,
            }
        ],
        "trainable_approximation_class": AvgPooling2d,
        "get_trainable_approximation_kwargs": [
            {
                "pooling": nn.MaxPool2d(
                    kernel_size,
                    stride=None,
                    padding=0,
                    dilation=1,
                    return_indices=False,
                    ceil_mode=False,
                ),
            }
        ],
        "pretrained_approximation_class": AvgPooling2d,
        "forward_kwargs": {
            "input": torch.arange(
                1, img_size * img_size + 1, device=DEVICE, dtype=float
            ).reshape(1, img_size, img_size),
        },
        "output_type": Tensor,
        "expected_output": [
            torch.ones((1, 1, 1), device=DEVICE, dtype=float) * 512.5,
        ],
    },
}


@pytest.mark.parametrize(
    "approximator_identifier,init_parameters",
    [
        (approx, init_params)
        for approx in list(testing_informations.keys())
        for init_params in testing_informations[approx]["init_parameters"]
    ],
    ids=[
        f"{approx} - init_params {init_params_index} "
        for approx in list(testing_informations.keys())
        for init_params_index, _ in enumerate(
            testing_informations[approx]["init_parameters"]
        )
    ],
)
def test_approximator_init(
    approximator_identifier: str, init_parameters: Dict[str, Any]
):
    """Tests the initialization of the approximator.

    Args:
        approximator_identifier: identifier of the approximator to be tested.
        init_parameters: initialization parameters of the approximation.
    """
    # retrieving the testing values for the approximator
    approximator_dictionary = testing_informations[approximator_identifier]

    # initializing the approximator object
    approximator = approximator_dictionary["approximator_class"](
        parameters=init_parameters
    )

    # ASSERTS

    # checking the approximator class
    assert isinstance(approximator, ModuleApproximator)
    assert isinstance(approximator, approximator_dictionary["approximator_class"])


@pytest.mark.parametrize(
    "approximator_identifier,init_parameters,init_parameters_index",
    [
        (approx, init_params, init_params_index)
        for approx in list(testing_informations.keys())
        for init_params_index, init_params in enumerate(
            testing_informations[approx]["init_parameters"]
        )
    ],
    ids=[
        f"{approx} - init_params {init_params_index} "
        for approx in list(testing_informations.keys())
        for init_params_index, _ in enumerate(
            testing_informations[approx]["init_parameters"]
        )
    ],
)
def test_get_trainable_approximation(
    approximator_identifier: str,
    init_parameters: Dict[str, Any],
    init_parameters_index: int,
):
    """Tests the approximator trainable approximation.

    Args:
        approximator_identifier: identifier of the approximator to be tested.
        init_parameters: initialization parameters of the approximation.
        init_parameters_index: index to select the testing information corresponding to the initialization parameters.
    """
    # retrieving the testing values for the approximator
    approximator_dictionary = testing_informations[approximator_identifier]

    # initializing the approximator object
    approximator = approximator_dictionary["approximator_class"](
        parameters=init_parameters
    )
    # getting the trainable approximation
    trainable_approx_module = approximator.get_trainable_approximation(
        **approximator_dictionary["get_trainable_approximation_kwargs"][
            init_parameters_index
        ]
    )

    # ASSERTS

    # checking the trainable approximation class
    assert isinstance(
        trainable_approx_module,
        approximator_dictionary["trainable_approximation_class"],
    )


@pytest.mark.parametrize(
    "approximator_identifier,init_parameters,init_parameters_index",
    [
        (approx, init_params, init_params_index)
        for approx in list(testing_informations.keys())
        for init_params_index, init_params in enumerate(
            testing_informations[approx]["init_parameters"]
        )
    ],
    ids=[
        f"{approx} - init_params {init_params_index} "
        for approx in list(testing_informations.keys())
        for init_params_index, _ in enumerate(
            testing_informations[approx]["init_parameters"]
        )
    ],
)
def test_trainable_approximation_forward(
    approximator_identifier: str,
    init_parameters: Dict[str, Any],
    init_parameters_index: int,
):
    """Tests the forward pass of the approximator trainable approximation.

    Args:
        approximator_identifier: identifier of the approximator to be tested.
        init_parameters: initialization parameters of the approximation.
        init_parameters_index: index to select the testing information corresponding to the initialization parameters.
    """
    # retrieving the testing values for the approximator
    approximator_dictionary = testing_informations[approximator_identifier]

    # initializing the approximator object
    approximator = approximator_dictionary["approximator_class"](
        parameters=init_parameters
    )
    # getting the trainable approximation
    trainable_approx_module = approximator.get_trainable_approximation(
        **approximator_dictionary["get_trainable_approximation_kwargs"][
            init_parameters_index
        ]
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
    if approximator_dictionary["expected_output"][init_parameters_index] is not None:
        assert torch.all(
            torch.eq(
                output,
                approximator_dictionary["expected_output"][init_parameters_index],
            )
        )


@pytest.mark.parametrize(
    "approximator_identifier,init_parameters,init_parameters_index",
    [
        (approx, init_params, init_params_index)
        for approx in list(testing_informations.keys())
        for init_params_index, init_params in enumerate(
            testing_informations[approx]["init_parameters"]
        )
    ],
    ids=[
        f"{approx} - init_params {init_params_index} "
        for approx in list(testing_informations.keys())
        for init_params_index, _ in enumerate(
            testing_informations[approx]["init_parameters"]
        )
    ],
)
def test_get_pretrained_approximation(
    approximator_identifier: str,
    init_parameters: Dict[str, Any],
    init_parameters_index: int,
):
    """Tests the approximator pretrained approximation.

    Args:
        approximator_identifier: identifier of the approximator to be tested.
        init_parameters: initialization parameters of the approximation.
        init_parameters_index: index to select the testing information corresponding to the initialization parameters.
    """
    # retrieving the testing values for the approximator
    approximator_dictionary = testing_informations[approximator_identifier]

    # initializing the approximator object
    approximator = approximator_dictionary["approximator_class"](
        parameters=init_parameters
    )
    # getting the trainable approximation
    trainable_approx_module = approximator.get_trainable_approximation(
        **approximator_dictionary["get_trainable_approximation_kwargs"][
            init_parameters_index
        ]
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
    "approximator_identifier,init_parameters,init_parameters_index",
    [
        (approx, init_params, init_params_index)
        for approx in list(testing_informations.keys())
        for init_params_index, init_params in enumerate(
            testing_informations[approx]["init_parameters"]
        )
    ],
    ids=[
        f"{approx} - init_params {init_params_index} "
        for approx in list(testing_informations.keys())
        for init_params_index, _ in enumerate(
            testing_informations[approx]["init_parameters"]
        )
    ],
)
def test_pretrained_approximation_forward(
    approximator_identifier: str,
    init_parameters: Dict[str, Any],
    init_parameters_index: int,
):
    """Tests the forward pass of the approximator pretrained approximation.

    Args:
        approximator_identifier: identifier of the approximator to be tested.
        init_parameters: initialization parameters of the approximation.
        init_parameters_index: index to select the testing information corresponding to the initialization parameters.
    """
    # retrieving the testing values for the approximator
    approximator_dictionary = testing_informations[approximator_identifier]

    # initializing the approximator object
    approximator = approximator_dictionary["approximator_class"](
        parameters=init_parameters
    )
    # getting the trainable approximation
    trainable_approx_module = approximator.get_trainable_approximation(
        **approximator_dictionary["get_trainable_approximation_kwargs"][
            init_parameters_index
        ]
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
    if approximator_dictionary["expected_output"][init_parameters_index] is not None:
        assert torch.all(
            torch.eq(
                output,
                approximator_dictionary["expected_output"][init_parameters_index],
            )
        )
