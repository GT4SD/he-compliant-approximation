"""Testing the approximation controller."""

from copy import deepcopy
from typing import Dict, Type, Union

import importlib_resources
import pytest
import torch
from torch import Tensor, nn

# importing modules' approximators and approximations
from hela.approximation.approximators.activation.quadratic import QuadraticApproximation
from hela.approximation.approximators.activation.trainable_quadratic import (
    PairedReLU,
    TrainableQuadraticApproximation,
)
from hela.approximation.approximators.attention.masking.multiplicative import (
    MultiplicativeAttentionMasking,
)
from hela.approximation.approximators.attention.query_key_product.not_scaled import (
    NotScaledQueryKeyDotProduct,
)
from hela.approximation.approximators.layer_normalization.batch_normalization import (
    BatchNorm1dForTransformers,
)
from hela.approximation.approximators.layer_normalization.distill_layernorm import (
    DistillLayerNormApproximation,
    PairedLayerNorm,
)
from hela.approximation.approximators.multihead.customizable_multihead import (
    CustomizableMultiHead,
)
from hela.approximation.approximators.pooling.avg_pooling_2d import (
    AvgPooling2dApproximation,
)
from hela.approximation.approximators.softmax.mlp_softmax import MLPSoftmaxApproximation
from hela.approximation.approximators.softmax.polynomial import PolynomialSoftmax
from hela.approximation.approximators.softmax.taylor import TaylorSoftmax

# importing the approximation controller
from hela.approximation.controller import ModelApproximationController, ToApproximate

# importing models and configurations classes
# LeNet
from hela.models.lenet.configuration import LeNetConfig
from hela.models.lenet.model import LeNet

# VanillaTransformer
from hela.models.vanilla_transformer.configuration import VanillaTransformerConfig
from hela.models.vanilla_transformer.model import (
    VanillaTransformer,
    VanillaTransformerOutput,
)

ALIASES_FILE = str(
    importlib_resources.files("hela") / "resources" / "approximation" / "aliases.json"
)

# default device to run the tests
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

###########################################
###### Model specific check functions #####
###########################################

#####################
# Vanilla Transformer
#####################

# defining some configuration parameters for the vanilla transformer
num_encoder_layers = num_decoder_layers = 4
embedding_dim = 256
ffnn_hidden_dim = 2048
num_attention_heads = 8


def check_relu_approximation_vanilla_transformer(
    approx_model: nn.Module,
    approximation_class: Type[nn.Module],
):
    """Checks if the substitution of the ReLU approximation have been performed correctly in a VanillaTransformer model.

    Args:
        approx_model: model with approximated modules.
        approximation_class: class of the approximation modules.
    """
    assert isinstance(approx_model, VanillaTransformer)
    for idx in range(num_encoder_layers):
        assert isinstance(
            approx_model.encoder.encoder.layers[idx].activation,
            approximation_class,
        ), "Wrong ReLU approximation for VanillaTransformer model"
    for idx in range(num_decoder_layers):
        assert isinstance(
            approx_model.decoder.decoder.layers[idx].activation,
            approximation_class,
        ), "Wrong ReLU approximation for VanillaTransformer model"


def check_layernorm_approximation_vanilla_transformer(
    approx_model: nn.Module,
    approximation_class: Type[nn.Module],
):
    """Checks if the substitution of the LayerNorm approximation have been performed correctly in a VanillaTransformer model.

    Args:
        approx_model: model with approximated modules.
        approximation_class: class of the approximation modules.
    """
    assert isinstance(approx_model, VanillaTransformer)
    for idx in range(num_encoder_layers):
        assert isinstance(
            approx_model.encoder.encoder.layers[idx].norm1, approximation_class
        ), "Wrong LayerNorm approximation for VanillaTransformer model"
        assert isinstance(
            approx_model.encoder.encoder.layers[idx].norm2, approximation_class
        ), "Wrong LayerNorm approximation for VanillaTransformer model"
    for idx in range(num_decoder_layers):
        assert isinstance(
            approx_model.decoder.decoder.layers[idx].norm1, approximation_class
        ), "Wrong LayerNorm approximation for VanillaTransformer model"
        assert isinstance(
            approx_model.decoder.decoder.layers[idx].norm2, approximation_class
        ), "Wrong LayerNorm approximation for VanillaTransformer model"
        assert isinstance(
            approx_model.decoder.decoder.layers[idx].norm3, approximation_class
        ), "Wrong LayerNorm approximation for VanillaTransformer model"


def check_multihead_approximation_vanilla_transformer(
    approx_model: nn.Module,
    approximation_class: Type[nn.Module],
):
    """Checks if the substitution of the MultiheadAttention approximation have been performed correctly in a VanillaTransformer model.

    Args:
        approx_model: model with approximated modules.
        approximation_class: class of the approximation modules.
    """
    assert isinstance(approx_model, VanillaTransformer)
    for idx in range(num_encoder_layers):
        assert isinstance(
            approx_model.encoder.encoder.layers[idx].self_attn,
            approximation_class,
        ), "Wrong MultiheadAttention approximation for VanillaTransformer model"
    for idx in range(num_decoder_layers):
        assert isinstance(
            approx_model.decoder.decoder.layers[idx].self_attn,
            approximation_class,
        ), "Wrong MultiheadAttention approximation for VanillaTransformer model"


def check_softmax_approximation_vanilla_transformer(
    approx_model: nn.Module,
    approximation_class: Type[nn.Module],
):
    """Checks if the substitution of the Softmax approximation have been performed correctly in a VanillaTransformer model.

    Args:
        approx_model: model with approximated modules.
        approximation_class: class of the approximation modules.
    """
    assert isinstance(approx_model, VanillaTransformer)
    for idx in range(num_encoder_layers):
        assert isinstance(
            approx_model.encoder.encoder.layers[idx].self_attn,
            CustomizableMultiHead,
        ), "Wrong MultiheadAttention approximation for VanillaTransformer model"
        assert isinstance(
            approx_model.encoder.encoder.layers[idx].self_attn.kernel_function,
            approximation_class,
        ), "Wrong Softmax approximation for VanillaTransformer model"
    for idx in range(num_decoder_layers):
        assert isinstance(
            approx_model.decoder.decoder.layers[idx].self_attn,
            CustomizableMultiHead,
        ), "Wrong MultiheadAttention approximation for VanillaTransformer model"
        assert isinstance(
            approx_model.decoder.decoder.layers[idx].self_attn.kernel_function,
            approximation_class,
        ), "Wrong Softmax approximation for VanillaTransformer model"


def check_attention_masking_approximation_vanilla_transformer(
    approx_model: nn.Module,
    approximation_class: Type[nn.Module],
):
    """Checks if the substitution of the attention masking approximation have been performed correctly in a VanillaTransformer model.

    Args:
        approx_model: model with approximated modules.
        approximation_class: class of the approximation modules.
    """
    assert isinstance(approx_model, VanillaTransformer)
    for idx in range(num_encoder_layers):
        assert isinstance(
            approx_model.encoder.encoder.layers[idx].self_attn,
            CustomizableMultiHead,
        ), "Wrong MultiheadAttention approximation for VanillaTransformer model"
        assert isinstance(
            approx_model.encoder.encoder.layers[idx].self_attn.attn_masking_function,
            approximation_class,
        ), "Wrong attention masking approximation for VanillaTransformer model"
    for idx in range(num_decoder_layers):
        assert isinstance(
            approx_model.decoder.decoder.layers[idx].self_attn,
            CustomizableMultiHead,
        ), "Wrong MultiheadAttention approximation for VanillaTransformer model"
        assert isinstance(
            approx_model.decoder.decoder.layers[idx].self_attn.attn_masking_function,
            approximation_class,
        ), "Wrong attention masking approximation for VanillaTransformer model"


def check_attention_query_key_product_approximation_vanilla_transformer(
    approx_model: nn.Module,
    approximation_class: Type[nn.Module],
):
    """Checks if the substitution of the query-key product approximation have been performed correctly in a VanillaTransformer model.

    Args:
        approx_model: model with approximated modules.
        approximation_class: class of the approximation modules.
    """
    assert isinstance(approx_model, VanillaTransformer)
    for idx in range(num_encoder_layers):
        assert isinstance(
            approx_model.encoder.encoder.layers[idx].self_attn,
            CustomizableMultiHead,
        ), "Wrong MultiheadAttention approximation for VanillaTransformer model"
        assert isinstance(
            approx_model.encoder.encoder.layers[idx].self_attn.query_key_product,
            approximation_class,
        ), "Wrong attention query-key dot product approximation for VanillaTransformer model"
    for idx in range(num_decoder_layers):
        assert isinstance(
            approx_model.decoder.decoder.layers[idx].self_attn,
            CustomizableMultiHead,
        ), "Wrong MultiheadAttention approximation for VanillaTransformer model"
        assert isinstance(
            approx_model.decoder.decoder.layers[idx].self_attn.query_key_product,
            approximation_class,
        ), "Wrong attention query-key dot product approximation for VanillaTransformer model"


#######
# LeNet
#######

# defining some configuration parameters for the lenet
lenet_type = "LeNet-5"
num_classes = 10
greyscale = True


def check_relu_approximation_lenet(
    approx_model: nn.Module,
    approximation_class: Type[nn.Module],
):
    """Checks if the substitution of the ReLU approximation have been performed correctly in a LeNet model.

    Args:
        approx_model: model with approximated modules.
        approximation_class: class of the approximation modules.
    """
    assert isinstance(approx_model, LeNet)

    assert isinstance(
        approx_model.layers[2], approximation_class
    ), "Wrong ReLU approximation for LeNet model"
    assert isinstance(
        approx_model.layers[6], approximation_class
    ), "Wrong ReLU approximation for LeNet model"
    assert isinstance(
        approx_model.layers[9], approximation_class
    ), "Wrong ReLU approximation for LeNet model"
    assert isinstance(
        approx_model.layers[11], approximation_class
    ), "Wrong ReLU approximation for LeNet model"


def check_maxpool2d_approximation_lenet(
    approx_model: nn.Module,
    approximation_class: Type[nn.Module],
):
    """Checks if the substitution of the max pooling 2d approximation have been performed correctly in a LeNet model.

    Args:
        approx_model: model with approximated modules.
        approximation_class: class of the approximation modules.
    """
    assert isinstance(approx_model, LeNet)
    assert isinstance(
        approx_model.layers[3], approximation_class
    ), "Wrong max pooling 2d approximation for LeNet model"
    assert isinstance(
        approx_model.layers[7], approximation_class
    ), "Wrong max pooling 2d approximation for LeNet model"


##############################
###### Test dictionary #######
##############################

# defining some testing parameters
batch_size = 2
sequence_length = 256
img_size = 32

# each item of the dictionary contains the information of a certain model on which you may test the approximation controller
model_testing_informations = {
    VanillaTransformer: {
        "config_class": VanillaTransformerConfig,
        "config_parameters": {
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "num_attention_heads": num_attention_heads,
            "ffn_hidden_dim": ffnn_hidden_dim,
            "embedding_dim": embedding_dim,
            "device": DEVICE,
        },
        "forward_input": {
            "encoder_input_ids": torch.ones(
                (batch_size, sequence_length), device=DEVICE
            ).long(),
            "decoder_input_ids": torch.ones(
                (batch_size, sequence_length), device=DEVICE
            ).long(),
        },
        "output_class": VanillaTransformerOutput,
    },
    LeNet: {
        "config_class": LeNetConfig,
        "config_parameters": {
            "lenet_type": lenet_type,
            "num_classes": num_classes,
            "greyscale": greyscale,
        },
        "forward_input": {
            "x": torch.ones((batch_size, 1, img_size, img_size), device=DEVICE)
        },
        "output_class": Tensor,
    },
}

# each dictionary entry represent a class of module to be approximated and contains its testing values
# for each 'to_approx_dict' a corresponding 'trainable_approximation_class' and 'pretrained_approximation_class' must be provided
approximation_testing_informations = {
    "ReluApproximation": {
        "model_classes": [VanillaTransformer, LeNet],
        "check_substitution": [
            check_relu_approximation_vanilla_transformer,
            check_relu_approximation_lenet,
        ],
        "to_approx_dict": [
            {
                "modules_set": [
                    {
                        "module": "relu",
                        "approximation_type": "quadratic",
                        "parameters": {},
                    },
                ]
            },
            {
                "modules_set": [
                    {
                        "module": "relu",
                        "approximation_type": "trainable_quadratic",
                        "parameters": {},
                    },
                ]
            },
        ],
        "trainable_approximation_class": [QuadraticApproximation, PairedReLU],
        "pretrained_approximation_class": [
            QuadraticApproximation,
            TrainableQuadraticApproximation,
        ],
        "default_approximation_class": QuadraticApproximation,
    },
    "LayerNormApproximation": {
        "model_classes": [VanillaTransformer],
        "check_substitution": [check_layernorm_approximation_vanilla_transformer],
        "to_approx_dict": [
            {
                "modules_set": [
                    {
                        "module": "layernorm",
                        "approximation_type": "batchnorm",
                        "parameters": {},
                    },
                ]
            },
            {
                "modules_set": [
                    {
                        "module": "layernorm",
                        "approximation_type": "distill_layernorm",
                        "parameters": {},
                    },
                ]
            },
        ],
        "trainable_approximation_class": [
            BatchNorm1dForTransformers,
            PairedLayerNorm,
        ],
        "pretrained_approximation_class": [
            BatchNorm1dForTransformers,
            DistillLayerNormApproximation,
        ],
        "default_approximation_class": BatchNorm1dForTransformers,
    },
    "MultiHeadApproximation": {
        "model_classes": [VanillaTransformer],
        "check_substitution": [check_multihead_approximation_vanilla_transformer],
        "to_approx_dict": [
            {
                "modules_set": [
                    {
                        "module": "multihead",
                        "approximation_type": "customizable_multihead",
                        "parameters": {},
                    },
                ]
            },
        ],
        "trainable_approximation_class": [
            CustomizableMultiHead,
        ],
        "pretrained_approximation_class": [
            CustomizableMultiHead,
        ],
        "default_approximation_class": CustomizableMultiHead,
    },
    "SoftmaxApproximation": {
        "model_classes": [VanillaTransformer],
        "check_substitution": [check_softmax_approximation_vanilla_transformer],
        "to_approx_dict": [
            {
                "modules_set": [
                    {
                        "module": "softmax",
                        "approximation_type": "polynomial",
                        "parameters": {},
                    },
                    {
                        "module": "multihead",
                        "approximation_type": "customizable_multihead",
                        "parameters": {},
                    },
                ]
            },
            {
                "modules_set": [
                    {
                        "module": "softmax",
                        "approximation_type": "taylor",
                        "parameters": {},
                    },
                    {
                        "module": "multihead",
                        "approximation_type": "customizable_multihead",
                        "parameters": {},
                    },
                ]
            },
            {
                "modules_set": [
                    {
                        "module": "softmax",
                        "approximation_type": "MLP_softmax",
                        "parameters": {"unit_test": True},
                    },
                    {
                        "module": "multihead",
                        "approximation_type": "customizable_multihead",
                        "parameters": {},
                    },
                ]
            },
        ],
        "trainable_approximation_class": [
            PolynomialSoftmax,
            TaylorSoftmax,
            MLPSoftmaxApproximation,
        ],
        "pretrained_approximation_class": [
            PolynomialSoftmax,
            TaylorSoftmax,
            MLPSoftmaxApproximation,
        ],
        "default_approximation_class": PolynomialSoftmax,
    },
    "AttentionMaskingApproximation": {
        "model_classes": [VanillaTransformer],
        "check_substitution": [
            check_attention_masking_approximation_vanilla_transformer
        ],
        "to_approx_dict": [
            {
                "modules_set": [
                    {
                        "module": "attn_masking",
                        "approximation_type": "multiplicative",
                        "parameters": {},
                    },
                    {
                        "module": "multihead",
                        "approximation_type": "customizable_multihead",
                        "parameters": {},
                    },
                ]
            },
        ],
        "trainable_approximation_class": [
            MultiplicativeAttentionMasking,
        ],
        "pretrained_approximation_class": [
            MultiplicativeAttentionMasking,
        ],
        "default_approximation_class": MultiplicativeAttentionMasking,
    },
    "AttentionQueryKeyProductApproximation": {
        "model_classes": [VanillaTransformer],
        "check_substitution": [
            check_attention_query_key_product_approximation_vanilla_transformer
        ],
        "to_approx_dict": [
            {
                "modules_set": [
                    {
                        "module": "query_key_product",
                        "approximation_type": "not_scaled",
                        "parameters": {},
                    },
                    {
                        "module": "multihead",
                        "approximation_type": "customizable_multihead",
                        "parameters": {},
                    },
                ]
            },
        ],
        "trainable_approximation_class": [
            NotScaledQueryKeyDotProduct,
        ],
        "pretrained_approximation_class": [
            NotScaledQueryKeyDotProduct,
        ],
        "default_approximation_class": NotScaledQueryKeyDotProduct,
    },
    "AvgPooling2dApproximation": {
        "model_classes": [LeNet],
        "check_substitution": [check_maxpool2d_approximation_lenet],
        "to_approx_dict": [
            {
                "modules_set": [
                    {
                        "module": "max_pooling_2d",
                        "approximation_type": "avg_pooling_2d",
                        "parameters": {},
                    },
                ]
            },
        ],
        "trainable_approximation_class": [
            AvgPooling2dApproximation,
        ],
        "pretrained_approximation_class": [
            AvgPooling2dApproximation,
        ],
        "default_approximation_class": AvgPooling2dApproximation,
    },
}


@pytest.mark.parametrize(
    "approximation_identifier,to_approx_dict",
    [
        (approx, init_params)
        for approx in list(approximation_testing_informations.keys())
        for init_params in approximation_testing_informations[approx]["to_approx_dict"]
    ],
    ids=[
        f"{approx} - to_approx_dict {to_approx_dict_index} "
        for approx in list(approximation_testing_informations.keys())
        for to_approx_dict_index, _ in enumerate(
            approximation_testing_informations[approx]["to_approx_dict"]
        )
    ],
)
def test_controller_init(
    approximation_identifier: str, to_approx_dict: Dict[str, Union[str, Dict]]
):
    """Tests the controller initialization.

    Args:
        approximation_identifier: identifier of the approximation to be tested.
        to_approx_dict: specification of the approximations to be performed.
    """
    # retrieving the testing values for the module to be approximated
    approximation_dictionary = approximation_testing_informations[
        approximation_identifier
    ]

    for model_class in approximation_dictionary["model_classes"]:
        model_dictionary = model_testing_informations[model_class]
        # initializing the model to approximate
        model = model_class(
            model_dictionary["config_class"](**model_dictionary["config_parameters"])
        )

        to_approximate = ToApproximate(**to_approx_dict)

        controller = ModelApproximationController(
            model=model,
            to_approximate=to_approximate,
            modules_aliases_file=ALIASES_FILE,
        )

        # ASSERTS

        assert isinstance(controller, ModelApproximationController)


@pytest.mark.parametrize(
    "approximation_identifier,to_approx_dict",
    [
        (approx, init_params)
        for approx in list(approximation_testing_informations.keys())
        for init_params in approximation_testing_informations[approx]["to_approx_dict"]
    ],
    ids=[
        f"{approx} - to_approx_dict {to_approx_dict_index} "
        for approx in list(approximation_testing_informations.keys())
        for to_approx_dict_index, _ in enumerate(
            approximation_testing_informations[approx]["to_approx_dict"]
        )
    ],
)
def test_controller_default_approximation_type_recover(
    approximation_identifier: str, to_approx_dict: Dict[str, Union[str, Dict]]
):
    """Tests the ability of the controller to use a default approximation if its type is not specified.

    Args:
        approximation_identifier: identifier of the approximation to be tested.
        to_approx_dict: specification of the approximations to be performed.
    """
    # retrieving the testing values for the module to be approximated
    approximation_dictionary = approximation_testing_informations[
        approximation_identifier
    ]

    for model_index, model_class in enumerate(
        approximation_dictionary["model_classes"]
    ):
        model_dictionary = model_testing_informations[model_class]

        # initializing the model to approximate
        model = model_class(
            model_dictionary["config_class"](**model_dictionary["config_parameters"])
        )

        # copying the testing data to avoid its modification
        approx_dict = deepcopy(to_approx_dict)
        for elem in approx_dict["modules_set"]:
            # the approximation_type is set to empty string (i.e. not specified), requiring the controller to set the default one
            elem["approximation_type"] = ""
            elem["parameters"] = {}

        # initializing the object containing the approximation to be handled by the controller
        to_approximate = ToApproximate(**approx_dict)
        # initializing the approximation controller
        controller = ModelApproximationController(
            model=model,
            to_approximate=to_approximate,
            modules_aliases_file=ALIASES_FILE,
        )
        # getting the trainable approximation of the model
        approx_model, num_subs_model = controller.get_approximated_model(
            pretrained=False, return_num_substitutions=True
        )
        num_subs_approx_model = controller.recursive_search_with_approximation(
            approx_model, pretrained=False
        )

        # ASSERTS

        # checking the model class
        assert isinstance(approx_model, nn.Module)
        assert isinstance(approx_model, model_class)

        # checking the number of substitution made to the model
        assert sum(num_subs_model.values()) > 0
        # checking the number of substitution made to the approximated model
        assert sum(num_subs_approx_model.values()) == 0

        # checking the class of the substituted modules inside the model
        approximation_dictionary["check_substitution"][model_index](
            approx_model,
            approximation_dictionary["default_approximation_class"],
        )


@pytest.mark.parametrize(
    "approximation_identifier,to_approx_dict,to_approx_dict_index",
    [
        (approx, init_params, init_params_index)
        for approx in list(approximation_testing_informations.keys())
        for init_params_index, init_params in enumerate(
            approximation_testing_informations[approx]["to_approx_dict"]
        )
    ],
    ids=[
        f"{approx} - to_approx_dict {to_approx_dict_index} "
        for approx in list(approximation_testing_informations.keys())
        for to_approx_dict_index, _ in enumerate(
            approximation_testing_informations[approx]["to_approx_dict"]
        )
    ],
)
def test_controller_trainable_approximation(
    approximation_identifier: str,
    to_approx_dict: Dict[str, Union[str, Dict]],
    to_approx_dict_index: int,
):
    """Tests the trainable approximation of the model.

    Args:
        approximation_identifier: identifier of the approximation to be tested.
        to_approx_dict: specification of the approximations to be performed.
        to_approx_dict_index: index to select the testing information corresponding to the approximations to be performed.
    """
    # retrieving the testing values for the module to be approximated
    approximation_dictionary = approximation_testing_informations[
        approximation_identifier
    ]

    for model_index, model_class in enumerate(
        approximation_dictionary["model_classes"]
    ):
        model_dictionary = model_testing_informations[model_class]

        # initializing the model to approximate
        model = model_class(
            model_dictionary["config_class"](**model_dictionary["config_parameters"])
        )
        # initializing the object containing the approximation to be handled by the controller
        to_approximate = ToApproximate(**to_approx_dict)
        # initializing the approximation controller
        controller = ModelApproximationController(
            model=model,
            to_approximate=to_approximate,
            modules_aliases_file=ALIASES_FILE,
        )
        # getting the trainable approximation of the model
        approx_model, num_subs_model = controller.get_approximated_model(
            pretrained=False, return_num_substitutions=True
        )
        # getting the number of substitution made to the approximated trainable model
        num_subs_approx_model = controller.recursive_search_with_approximation(
            approx_model, pretrained=False
        )

        # ASSERTS

        # checking the model class
        assert isinstance(approx_model, nn.Module)
        assert isinstance(approx_model, model_class)
        # checking the number of substitution made to the model
        assert sum(num_subs_model.values()) > 0
        # checking the number of substitution made to the approximated trainable model
        assert sum(num_subs_approx_model.values()) == 0

        # checking the class of the substituted modules inside the model
        approximation_dictionary["check_substitution"][model_index](
            approx_model,
            approximation_dictionary["trainable_approximation_class"][
                to_approx_dict_index
            ],
        )


@pytest.mark.parametrize(
    "approximation_identifier,to_approx_dict",
    [
        (approx, init_params)
        for approx in list(approximation_testing_informations.keys())
        for init_params in approximation_testing_informations[approx]["to_approx_dict"]
    ],
    ids=[
        f"{approx} - to_approx_dict {to_approx_dict_index} "
        for approx in list(approximation_testing_informations.keys())
        for to_approx_dict_index, _ in enumerate(
            approximation_testing_informations[approx]["to_approx_dict"]
        )
    ],
)
def test_controller_trainable_approximation_forward(
    approximation_identifier: str,
    to_approx_dict: Dict[str, Union[str, Dict]],
):
    """Tests the forward pass of the trainable approximation of the model.

    Args:
        approximation_identifier: identifier of the approximation to be tested.
        to_approx_dict: specification of the approximations to be performed.
    """
    # retrieving the testing values for the module to be approximated
    approximation_dictionary = approximation_testing_informations[
        approximation_identifier
    ]

    for model_index, model_class in enumerate(
        approximation_dictionary["model_classes"]
    ):
        model_dictionary = model_testing_informations[model_class]

        # initializing the model to approximate
        model = model_class(
            model_dictionary["config_class"](**model_dictionary["config_parameters"])
        )
        # initializing the object containing the approximation to be handled by the controller
        to_approximate = ToApproximate(**to_approx_dict)
        # initializing the approximation controller
        controller = ModelApproximationController(
            model=model,
            to_approximate=to_approximate,
            modules_aliases_file=ALIASES_FILE,
        )
        # getting the trainable approximation of the model
        approx_model = controller.get_approximated_model(pretrained=False)
        # moving the approximated model to the default DEVICE
        approx_model.to(DEVICE)

        # forward pass
        output = approx_model(**model_dictionary["forward_input"])

        # ASSERTS

        # checking the model class
        assert isinstance(approx_model, nn.Module)
        assert isinstance(approx_model, model_class)
        # checking the output class
        assert isinstance(output, model_dictionary["output_class"])


@pytest.mark.parametrize(
    "approximation_identifier,to_approx_dict,to_approx_dict_index",
    [
        (approx, init_params, init_params_index)
        for approx in list(approximation_testing_informations.keys())
        for init_params_index, init_params in enumerate(
            approximation_testing_informations[approx]["to_approx_dict"]
        )
    ],
    ids=[
        f"{approx} - to_approx_dict {to_approx_dict_index} "
        for approx in list(approximation_testing_informations.keys())
        for to_approx_dict_index, _ in enumerate(
            approximation_testing_informations[approx]["to_approx_dict"]
        )
    ],
)
def test_controller_pretrained_approximation(
    approximation_identifier: str,
    to_approx_dict: Dict[str, Union[str, Dict]],
    to_approx_dict_index: int,
):
    """Tests the pretrained approximation of the model.

    Args:
        approximation_identifier: identifier of the approximation to be tested.
        to_approx_dict: specification of the approximations to be performed.
        to_approx_dict_index: index to select the testing information corresponding to the approximations to be performed.
    """
    # retrieving the testing values for the module to be approximated
    approximation_dictionary = approximation_testing_informations[
        approximation_identifier
    ]

    for model_index, model_class in enumerate(
        approximation_dictionary["model_classes"]
    ):
        model_dictionary = model_testing_informations[model_class]

        # initializing the model to approximate
        model = model_class(
            model_dictionary["config_class"](**model_dictionary["config_parameters"])
        )

        # initializing the object containing the approximation to be handled by the controller
        to_approximate = ToApproximate(**to_approx_dict)
        # initializing the approximation controller
        controller = ModelApproximationController(
            model=model,
            to_approximate=to_approximate,
            modules_aliases_file=ALIASES_FILE,
        )
        # getting the trainable approximation of the model
        approx_model, num_subs_model = controller.get_approximated_model(
            pretrained=False, return_num_substitutions=True
        )
        # getting the pretrained approximation of the model
        approx_model = controller.get_approximated_model(pretrained=True)

        # getting the number of substitution made to the approximated pretrained model
        num_subs_approx_model = controller.recursive_search_with_approximation(
            approx_model, pretrained=False
        )

        # ASSERTS

        # checking the model class
        assert isinstance(approx_model, nn.Module)
        assert isinstance(approx_model, model_class)
        # checking the number of substitution made to the model
        assert sum(num_subs_model.values()) > 0
        # checking the number of substitution made to the approximated pretrained model
        assert sum(num_subs_approx_model.values()) == 0

        # checking the class of the substituted modules inside the model
        approximation_dictionary["check_substitution"][model_index](
            approx_model,
            approximation_dictionary["pretrained_approximation_class"][
                to_approx_dict_index
            ],
        )


@pytest.mark.parametrize(
    "approximation_identifier,to_approx_dict",
    [
        (approx, init_params)
        for approx in list(approximation_testing_informations.keys())
        for init_params in approximation_testing_informations[approx]["to_approx_dict"]
    ],
    ids=[
        f"{approx} - to_approx_dict {to_approx_dict_index} "
        for approx in list(approximation_testing_informations.keys())
        for to_approx_dict_index, _ in enumerate(
            approximation_testing_informations[approx]["to_approx_dict"]
        )
    ],
)
def test_controller_pretrained_approximation_forward(
    approximation_identifier: str, to_approx_dict: Dict[str, Union[str, Dict]]
):
    """Tests the forward pass of the pretrained approximation of the model.

    Args:
        approximation_identifier: identifier of the approximation to be tested.
        to_approx_dict: specification of the approximations to be performed.
    """
    # retrieving the testing values for the module to be approximated
    approximation_dictionary = approximation_testing_informations[
        approximation_identifier
    ]

    for model_index, model_class in enumerate(
        approximation_dictionary["model_classes"]
    ):
        model_dictionary = model_testing_informations[model_class]

        # initializing the model to approximate
        model = model_class(
            model_dictionary["config_class"](**model_dictionary["config_parameters"])
        )
        # initializing the object containing the approximation to be handled by the controller
        to_approximate = ToApproximate(**to_approx_dict)
        # initializing the approximation controller
        controller = ModelApproximationController(
            model=model,
            to_approximate=to_approximate,
            modules_aliases_file=ALIASES_FILE,
        )
        # getting the trainable approximation of the model
        approx_model = controller.get_approximated_model(pretrained=False)
        # getting the pretrained approximation of the model
        approx_model = controller.get_approximated_model(pretrained=True)
        # moving the approximated model to the default DEVICE
        approx_model.to(DEVICE)

        # forward pass
        output = approx_model(**model_dictionary["forward_input"])

        # ASSERTS

        # checking the model class
        assert isinstance(approx_model, nn.Module)
        assert isinstance(approx_model, model_class)
        # checking the output class
        assert isinstance(output, model_dictionary["output_class"])
