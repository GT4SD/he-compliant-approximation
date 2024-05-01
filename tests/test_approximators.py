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

# defining the devices to run the tests on
DEVICE_LIST = ["cpu", "cuda"]

# defining some testing parameters
BATCH_SIZE = 10
SEQUENCE_LENGTH = EMBEDDING_DIM = 256

KERNEL_SIZE = 32
IMAGE_SIZE = 32


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
            "input": torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH)) * 2
        },
        "output_type": Tensor,
        "expected_output": [
            torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH)) * 4,
        ],
    },
    "TrainableQuadraticApproximator": {
        "approximator_class": TrainableQuadraticApproximator,
        "init_parameters": [{"input_dimension": SEQUENCE_LENGTH}],
        "trainable_approximation_class": PairedReLU,
        "get_trainable_approximation_kwargs": [{}],
        "pretrained_approximation_class": TrainableQuadraticActivation,
        "forward_kwargs": {
            "input": torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH)) * 2
        },
        "output_type": Tensor,
        "expected_output": [
            torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH)) * 2,
        ],
    },
    "LayerNormToBatchNormApproximator": {
        "approximator_class": LayerNormToBatchNormApproximator,
        "init_parameters": [{"num_features": EMBEDDING_DIM}],
        "trainable_approximation_class": BatchNorm1dForTransformers,
        "get_trainable_approximation_kwargs": [{}],
        "pretrained_approximation_class": BatchNorm1dForTransformers,
        "forward_kwargs": {
            "input": torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIM)) * 2
        },
        "output_type": Tensor,
        "expected_output": [
            None,
        ],
    },
    "DistillLayernormApproximator": {
        "approximator_class": DistillLayerNormApproximator,
        "init_parameters": [{"normalized_shape": EMBEDDING_DIM}],
        "trainable_approximation_class": PairedLayerNorm,
        "get_trainable_approximation_kwargs": [
            {"layernorm": nn.LayerNorm(EMBEDDING_DIM)}
        ],
        "pretrained_approximation_class": DistillLayerNorm,
        "forward_kwargs": {
            "input": torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIM)) * 2
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
                "embed_dim": EMBEDDING_DIM,
                "num_heads": int(EMBEDDING_DIM / 2),
                "dropout": 0.5,
            },
        ],
        "trainable_approximation_class": CustomizableMultiHead,
        "get_trainable_approximation_kwargs": [
            {
                "multihead": nn.MultiheadAttention(
                    embed_dim=EMBEDDING_DIM, num_heads=1
                ),
            },
            {
                "multihead": nn.MultiheadAttention(
                    embed_dim=EMBEDDING_DIM, num_heads=1
                ),
            },
        ],
        "pretrained_approximation_class": CustomizableMultiHead,
        "forward_kwargs": {
            "query": torch.ones((BATCH_SIZE, EMBEDDING_DIM)),
            "key": torch.ones((BATCH_SIZE, EMBEDDING_DIM)),
            "value": torch.ones((BATCH_SIZE, EMBEDDING_DIM)),
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
            "input": torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH)) * 2,
        },
        "output_type": Tensor,
        "expected_output": [
            torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH))
            / SEQUENCE_LENGTH,
            torch.pow(
                torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH)) * 2,
                2,
            ),
            torch.pow(
                torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH)) * 2,
                4,
            ),
            torch.pow(
                torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH)) * 2,
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
            "input": torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH)),
        },
        "output_type": Tensor,
        "expected_output": [
            torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH))
            / SEQUENCE_LENGTH,
        ],
    },
    "MLPSoftmaxApproximator": {
        "approximator_class": MLPSoftmaxApproximator,
        "init_parameters": [{"dim_size": EMBEDDING_DIM, "unit_test": True}],
        "trainable_approximation_class": MLPSoftmax,
        "get_trainable_approximation_kwargs": [{}],
        "pretrained_approximation_class": MLPSoftmax,
        "forward_kwargs": {
            "input": torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH)),
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
            "attn": torch.ones((SEQUENCE_LENGTH, SEQUENCE_LENGTH)),
            "attn_mask": torch.triu(
                torch.ones((SEQUENCE_LENGTH, SEQUENCE_LENGTH)) * float("-inf"),
                diagonal=1,
            ),
        },
        "output_type": Tensor,
        "expected_output": [
            torch.tril(
                torch.ones((SEQUENCE_LENGTH, SEQUENCE_LENGTH)),
                diagonal=0,
            ),
            torch.ones((SEQUENCE_LENGTH, SEQUENCE_LENGTH)),
        ],
    },
    "NotScaledQueryKeyProductApproximator": {
        "approximator_class": NotScaledQueryKeyDotProductApproximator,
        "init_parameters": [{}],
        "trainable_approximation_class": NotScaledQueryKeyDotProduct,
        "get_trainable_approximation_kwargs": [{}],
        "pretrained_approximation_class": NotScaledQueryKeyDotProduct,
        "forward_kwargs": {
            "query": torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIM)),
            "key": torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_DIM)) * 2,
        },
        "output_type": Tensor,
        "expected_output": [
            torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH))
            * 2
            * EMBEDDING_DIM,
        ],
    },
    "AvgPooling2dApproximator": {
        "approximator_class": AvgPooling2dApproximator,
        "init_parameters": [
            {
                "kernel_size": KERNEL_SIZE,
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
                    KERNEL_SIZE,
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
            "input": torch.arange(1, IMAGE_SIZE * IMAGE_SIZE + 1, dtype=float).reshape(
                1, IMAGE_SIZE, IMAGE_SIZE
            ),
        },
        "output_type": Tensor,
        "expected_output": [
            torch.ones((1, 1, 1), dtype=float) * 512.5,
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
        f" {approx} - init_params {init_params_index} "
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
    assert isinstance(
        approximator, ModuleApproximator
    ), "The created approximator is not an instance of ModuleApproximator."
    assert isinstance(
        approximator, approximator_dictionary["approximator_class"]
    ), f"The created approximator is not an instance of the expected class {approximator_dictionary['approximator_class'].__name__}."


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
        f" {approx} - init_params {init_params_index} "
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
    ), f"Trainable approximation module should be an instance of {approximator_dictionary['trainable_approximation_class'].__name__}"


@pytest.mark.parametrize(
    "approximator_identifier,init_parameters,init_parameters_index,device",
    [
        (approx, init_params, init_params_index, device)
        for approx in list(testing_informations.keys())
        for init_params_index, init_params in enumerate(
            testing_informations[approx]["init_parameters"]
        )
        for device in DEVICE_LIST
    ],
    ids=[
        f" {approx} - init_params {init_params_index} - device: {device} "
        for approx in list(testing_informations.keys())
        for init_params_index, _ in enumerate(
            testing_informations[approx]["init_parameters"]
        )
        for device in DEVICE_LIST
    ],
)
def test_trainable_approximation_forward(
    approximator_identifier: str,
    init_parameters: Dict[str, Any],
    init_parameters_index: int,
    device: str,
):
    """Tests the forward pass of the approximator trainable approximation.

    Args:
        approximator_identifier: identifier of the approximator to be tested.
        init_parameters: initialization parameters of the approximation.
        init_parameters_index: index to select the testing information corresponding to the initialization parameters.
        device: device to be used for the forward pass.
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
    # moving the approximated module to the device
    trainable_approx_module.to(device)
    # initializing the forward input
    forward_kwargs = deepcopy(approximator_dictionary["forward_kwargs"])
    for key, value in forward_kwargs.items():
        print(type(value))
        if isinstance(value, Tensor):
            forward_kwargs[key] = value.to(device)
    # forward pass through the trainable approximation
    output = trainable_approx_module(**forward_kwargs)

    # ASSERTS

    # checking the output type
    assert isinstance(
        output, approximator_dictionary["output_type"]
    ), f"Output type mismatch. Expected type: {approximator_dictionary['output_type']}."

    # checking the output values if they are specified
    if approximator_dictionary["expected_output"][
        init_parameters_index
    ] is not None and isinstance(
        approximator_dictionary["expected_output"][init_parameters_index], Tensor
    ):
        assert torch.all(
            torch.eq(
                output,
                approximator_dictionary["expected_output"][init_parameters_index].to(
                    device
                ),
            )
        ), "Output values do not match the expected values."

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()


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
        f" {approx} - init_params {init_params_index} "
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
    ), f"Trainable approximation module is not an instance of {approximator_dictionary['trainable_approximation_class'].__name__}"
    # checking the pretrained approximation class
    assert isinstance(
        pretrained_approx_module,
        approximator_dictionary["pretrained_approximation_class"],
    ), f"Pretrained approximation module is not an instance of {approximator_dictionary['pretrained_approximation_class'].__name__}"


@pytest.mark.parametrize(
    "approximator_identifier,init_parameters,init_parameters_index,device",
    [
        (approx, init_params, init_params_index, device)
        for approx in list(testing_informations.keys())
        for init_params_index, init_params in enumerate(
            testing_informations[approx]["init_parameters"]
        )
        for device in DEVICE_LIST
    ],
    ids=[
        f" {approx} - init_params {init_params_index} - device: {device} "
        for approx in list(testing_informations.keys())
        for init_params_index, _ in enumerate(
            testing_informations[approx]["init_parameters"]
        )
        for device in DEVICE_LIST
    ],
)
def test_pretrained_approximation_forward(
    approximator_identifier: str,
    init_parameters: Dict[str, Any],
    init_parameters_index: int,
    device: str,
):
    """Tests the forward pass of the approximator pretrained approximation.

    Args:
        approximator_identifier: identifier of the approximator to be tested.
        init_parameters: initialization parameters of the approximation.
        init_parameters_index: index to select the testing information corresponding to the initialization parameters.
        device: device to be used for the forward pass.
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
    # moving the approximated module to the device
    pretrained_approx_module.to(device)
    # initializing the forward input
    forward_kwargs = deepcopy(approximator_dictionary["forward_kwargs"])
    for key, value in forward_kwargs.items():
        if isinstance(value, torch.Tensor):
            forward_kwargs[key] = value.to(device)
    # forward pass through the trainable approximation
    output = pretrained_approx_module(**forward_kwargs)

    # ASSERTS

    # checking the output type
    assert isinstance(
        output, approximator_dictionary["output_type"]
    ), f"Output type mismatch. Expected type: {approximator_dictionary['output_type']}."

    # checking the output values if they are specified
    if approximator_dictionary["expected_output"][
        init_parameters_index
    ] is not None and isinstance(
        approximator_dictionary["expected_output"][init_parameters_index], Tensor
    ):
        assert torch.all(
            torch.eq(
                output,
                approximator_dictionary["expected_output"][init_parameters_index].to(
                    device
                ),
            )
        ), "Output values do not match the expected values."

    # releasing GPU memory, if needed
    if device == "cuda":
        torch.cuda.empty_cache()
