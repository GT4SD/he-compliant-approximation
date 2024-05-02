"""Testing the approximation controller."""

from copy import deepcopy
from typing import Dict, Type, Union

import importlib_resources
import pytest
import torch
from torch import Tensor, nn

# importing modules' approximators and approximations
from hela.approximation.approximators.activation.quadratic import QuadraticActivation
from hela.approximation.approximators.activation.trainable_quadratic import (
    PairedReLU,
    TrainableQuadraticActivation,
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
    DistillLayerNorm,
    PairedLayerNorm,
)
from hela.approximation.approximators.multihead.customizable_multihead import (
    CustomizableMultiHead,
)
from hela.approximation.approximators.pooling.avg_pooling_2d import AvgPooling2d
from hela.approximation.approximators.softmax.mlp_softmax import MLPSoftmax
from hela.approximation.approximators.softmax.polynomial import PolynomialSoftmax
from hela.approximation.approximators.softmax.taylor import TaylorSoftmax

# importing the approximation controller
from hela.approximation.controller import ModelApproximationController, ToApproximate

# importing models and configurations classes
#############################################
# AlexNet
from hela.models.alexnet.configuration import AlexNetConfig
from hela.models.alexnet.model import AlexNet

# LeNet
from hela.models.lenet.configuration import LeNetConfig
from hela.models.lenet.model import LeNet

# SqueezeNet
from hela.models.squeezenet.configuration import SqueezeNetConfig
from hela.models.squeezenet.model import SqueezeNet

# VanillaTransformer
from hela.models.vanilla_transformer.configuration import VanillaTransformerConfig
from hela.models.vanilla_transformer.model import (
    VanillaTransformer,
    VanillaTransformerOutput,
)

#############################################

ALIASES_FILE = str(
    importlib_resources.files("hela") / "resources" / "approximation" / "aliases.json"
)

# defining the devices to run the tests on
DEVICE_LIST = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

#########################################################
###### Model specific modules' approximation checks #####
#########################################################

#######################
# Vanilla Transformer #
#######################

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
    assert isinstance(
        approx_model, VanillaTransformer
    ), f"Wrong model type: {type(approx_model)}."
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
    assert isinstance(
        approx_model, VanillaTransformer
    ), f"Wrong model type: {type(approx_model)}."
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
    assert isinstance(
        approx_model, VanillaTransformer
    ), f"Wrong model type: {type(approx_model)}."
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
    assert isinstance(
        approx_model, VanillaTransformer
    ), f"Wrong model type: {type(approx_model)}."
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
    assert isinstance(
        approx_model, VanillaTransformer
    ), f"Wrong model type: {type(approx_model)}."
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
    assert isinstance(
        approx_model, VanillaTransformer
    ), f"Wrong model type: {type(approx_model)}."
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


#########
# LeNet #
#########

# defining some configuration parameters for the lenet
lenet_type = "lenet-5"
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
    assert isinstance(approx_model, LeNet), f"Wrong model type: {type(approx_model)}."

    relu_indexes = [2, 6, 9, 11]

    for idx in relu_indexes:
        assert isinstance(
            approx_model.layers[idx], approximation_class
        ), f"Wrong ReLU approximation for LeNet model at index {idx}."


def check_maxpool2d_approximation_lenet(
    approx_model: nn.Module,
    approximation_class: Type[nn.Module],
):
    """Checks if the substitution of the max pooling 2d approximation have been performed correctly in a LeNet model.

    Args:
        approx_model: model with approximated modules.
        approximation_class: class of the approximation modules.
    """
    assert isinstance(approx_model, LeNet), f"Wrong model type: {type(approx_model)}."

    maxpool2d_indexes = [3, 7]

    for idx in maxpool2d_indexes:
        assert isinstance(
            approx_model.layers[idx], approximation_class
        ), f"Wrong max pooling 2d approximation for LeNet model at index {idx}."


###########
# AlexNet #
###########

# defining some configuration parameters for the AlexNet
num_classes = 10
dropout = 0.5


def check_relu_approximation_alexnet(
    approx_model: nn.Module,
    approximation_class: Type[nn.Module],
):
    """Checks if the substitution of the ReLU approximation have been performed correctly in a AlexNet model.

    Args:
        approx_model: model with approximated modules.
        approximation_class: class of the approximation modules.
    """
    assert isinstance(
        approx_model, AlexNet
    ), f"Wrong model type: {type(approx_model)} ."

    features_relu_indexes = [1, 4, 7, 9, 11]
    classifier_relu_indexes = [2, 5]

    for idx in features_relu_indexes:
        assert isinstance(
            approx_model.model.features[idx], approximation_class
        ), f"Wrong ReLU approximation for AlexNet model inside the features section at index {idx}."

    for idx in classifier_relu_indexes:
        assert isinstance(
            approx_model.model.classifier[idx], approximation_class
        ), f"Wrong ReLU approximation for AlexNet model inside the classifier section at index {idx}."


def check_maxpool2d_approximation_alexnet(
    approx_model: nn.Module,
    approximation_class: Type[nn.Module],
):
    """Checks if the substitution of the max pooling 2d approximation have been performed correctly in a AlexNet model.

    Args:
        approx_model: model with approximated modules.
        approximation_class: class of the approximation modules.
    """
    assert isinstance(
        approx_model, AlexNet
    ), f"Wrong model type: {type(approx_model)} ."

    maxpool2d_indexes = [2, 5, 12]

    for idx in maxpool2d_indexes:
        assert isinstance(
            approx_model.model.features[idx], approximation_class
        ), f"Wrong max pooling 2d approximation for AlexNet model inside the features section at index {idx}."


##############
# SqueezeNet #
##############

# defining some configuration parameters for the Squeezenet
squeezenet_version = "1_0"
num_classes = 10
dropout = 0.5


def check_relu_approximation_squeezenet(
    approx_model: nn.Module,
    approximation_class: Type[nn.Module],
):
    """Checks if the substitution of the ReLU approximation have been performed correctly in a SqueezeNet model.

    Args:
        approx_model: model with approximated modules.
        approximation_class: class of the approximation modules.
    """
    assert isinstance(
        approx_model, SqueezeNet
    ), f"Wrong model type: {type(approx_model)} ."

    features_relu_indexes = [1]
    fire_blocks_indexes = [3, 4, 5, 7, 8, 9, 10, 12]
    fire_relu_keys = [
        "squeeze_activation",
        "expand1x1_activation",
        "expand3x3_activation",
    ]
    classifier_relu_indexes = [2]

    for idx in features_relu_indexes:
        assert isinstance(
            approx_model.model.features[idx], approximation_class
        ), f"Wrong ReLU approximation for SqueezeNet model inside the features section at index {idx}."

    for idx in fire_blocks_indexes:
        for key in fire_relu_keys:
            assert isinstance(
                getattr(approx_model.model.features[idx], key), approximation_class
            ), f"Wrong ReLU approximation for SqueezeNet model inside the fire block ({key}) at index {idx} in the features section."

    for idx in classifier_relu_indexes:
        assert isinstance(
            approx_model.model.classifier[idx], approximation_class
        ), f"Wrong ReLU approximation for SqueezeNet model inside the classifier section at index {idx}."


def check_maxpool2d_approximation_squeezenet(
    approx_model: nn.Module,
    approximation_class: Type[nn.Module],
):
    """Checks if the substitution of the max pooling 2d approximation have been performed correctly in a SqueezeNet model.

    Args:
        approx_model: model with approximated modules.
        approximation_class: class of the approximation modules.
    """
    assert isinstance(
        approx_model, SqueezeNet
    ), f"Wrong model type: {type(approx_model)} ."

    maxpool2d_indexes = [2, 6, 11]

    for idx in maxpool2d_indexes:
        assert isinstance(
            approx_model.model.features[idx], approximation_class
        ), f"Wrong max pooling 2d approximation for SqueezeNet model inside the features section at index {idx}."


############################################
###### Tests' information dictionary #######
############################################

# defining some testing parameters
BATCH_SIZE = 2
SEQUENCE_LENGTH = 256
IMG_SIZE = 32

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
        },
        "forward_input": {
            "encoder_input_ids": torch.ones((BATCH_SIZE, SEQUENCE_LENGTH)).long(),
            "decoder_input_ids": torch.ones((BATCH_SIZE, SEQUENCE_LENGTH)).long(),
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
        "forward_input": {"x": torch.ones((BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE))},
        "output_class": Tensor,
    },
    AlexNet: {
        "config_class": AlexNetConfig,
        "config_parameters": {
            "num_classes": num_classes,
            "dropout": dropout,
        },
        "forward_input": {"x": torch.ones((BATCH_SIZE, 3, 224, 224))},
        "output_class": Tensor,
    },
    SqueezeNet: {
        "config_class": SqueezeNetConfig,
        "config_parameters": {
            "version": squeezenet_version,
            "num_classes": num_classes,
            "dropout": dropout,
        },
        "forward_input": {"x": torch.ones((BATCH_SIZE, 3, 224, 224))},
        "output_class": Tensor,
    },
}

# each dictionary entry represent a class of module to be approximated and contains its testing values
# for each 'to_approx_dict' a corresponding 'trainable_approximation_class' and 'pretrained_approximation_class' must be provided
approximation_testing_informations = {
    "ReluApproximation": {
        "model_classes": [VanillaTransformer, LeNet, AlexNet, SqueezeNet],
        "check_substitution": [
            check_relu_approximation_vanilla_transformer,
            check_relu_approximation_lenet,
            check_relu_approximation_alexnet,
            check_relu_approximation_squeezenet,
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
        "trainable_approximation_class": [QuadraticActivation, PairedReLU],
        "pretrained_approximation_class": [
            QuadraticActivation,
            TrainableQuadraticActivation,
        ],
        "default_approximation_class": QuadraticActivation,
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
            DistillLayerNorm,
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
            MLPSoftmax,
        ],
        "pretrained_approximation_class": [
            PolynomialSoftmax,
            TaylorSoftmax,
            MLPSoftmax,
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
    "AvgPooling2d": {
        "model_classes": [LeNet, AlexNet, SqueezeNet],
        "check_substitution": [
            check_maxpool2d_approximation_lenet,
            check_maxpool2d_approximation_alexnet,
            check_maxpool2d_approximation_squeezenet,
        ],
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
            AvgPooling2d,
        ],
        "pretrained_approximation_class": [
            AvgPooling2d,
        ],
        "default_approximation_class": AvgPooling2d,
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
        f" {approx} - to_approx_dict {to_approx_dict_index} "
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
        f" {approx} - to_approx_dict {to_approx_dict_index} "
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
        f" {approx} - to_approx_dict {to_approx_dict_index} "
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
        assert isinstance(
            approx_model, nn.Module
        ), "The approximated model is not an instance of nn.Module."
        assert isinstance(
            approx_model, model_class
        ), f"The approximated model is not an instance of the expected model class {model_class.__name__}."
        # checking the number of substitution made to the model
        assert (
            sum(num_subs_model.values()) > 0
        ), "No substitutions were made to the model."
        # checking the number of substitution made to the approximated trainable model
        assert (
            sum(num_subs_approx_model.values()) == 0
        ), "Substitutions were unexpectedly made to the approximated trainable model."

        # checking the class of the substituted modules inside the model
        approximation_dictionary["check_substitution"][model_index](
            approx_model,
            approximation_dictionary["trainable_approximation_class"][
                to_approx_dict_index
            ],
        )


@pytest.mark.parametrize(
    "approximation_identifier,to_approx_dict,device",
    [
        (approx, init_params, device)
        for approx in list(approximation_testing_informations.keys())
        for init_params in approximation_testing_informations[approx]["to_approx_dict"]
        for device in DEVICE_LIST
    ],
    ids=[
        f" {approx} - to_approx_dict {to_approx_dict_index} - device: {device} "
        for approx in list(approximation_testing_informations.keys())
        for to_approx_dict_index, _ in enumerate(
            approximation_testing_informations[approx]["to_approx_dict"]
        )
        for device in DEVICE_LIST
    ],
)
def test_controller_trainable_approximation_forward(
    approximation_identifier: str,
    to_approx_dict: Dict[str, Union[str, Dict]],
    device: str,
):
    """Tests the forward pass of the trainable approximation of the model.

    Args:
        approximation_identifier: identifier of the approximation to be tested.
        to_approx_dict: specification of the approximations to be performed.
        device: device on which the model will be tested.
    """
    # retrieving the testing values for the module to be approximated
    approximation_dictionary = approximation_testing_informations[
        approximation_identifier
    ]

    for model_index, model_class in enumerate(
        approximation_dictionary["model_classes"]
    ):
        model_dictionary = model_testing_informations[model_class]

        config_parameters = model_dictionary["config_parameters"]
        if model_class == VanillaTransformer:
            config_parameters["device"] = device

        # initializing the model to approximate
        model = model_class(model_dictionary["config_class"](**config_parameters))
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
        # moving the approximated model to the specified device
        approx_model.to(device)
        # initializing the forward input
        forward_input = deepcopy(model_dictionary["forward_input"])
        for key, value in forward_input.items():
            if isinstance(value, Tensor):
                forward_input[key] = value.to(device)
        # forward pass
        output = approx_model(**forward_input)

        # ASSERTS

        # checking the model class
        assert isinstance(approx_model, nn.Module)
        assert isinstance(approx_model, model_class)
        # checking the output class
        assert isinstance(output, model_dictionary["output_class"])

        # releasing GPU memory, if needed
        if device == "cuda":
            torch.cuda.empty_cache()


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
        f" {approx} - to_approx_dict {to_approx_dict_index} "
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
    "approximation_identifier,to_approx_dict,device",
    [
        (approx, init_params, device)
        for approx in list(approximation_testing_informations.keys())
        for init_params in approximation_testing_informations[approx]["to_approx_dict"]
        for device in DEVICE_LIST
    ],
    ids=[
        f" {approx} - to_approx_dict {to_approx_dict_index} - device: {device}"
        for approx in list(approximation_testing_informations.keys())
        for to_approx_dict_index, _ in enumerate(
            approximation_testing_informations[approx]["to_approx_dict"]
        )
        for device in DEVICE_LIST
    ],
)
def test_controller_pretrained_approximation_forward(
    approximation_identifier: str,
    to_approx_dict: Dict[str, Union[str, Dict]],
    device: str,
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

        config_parameters = model_dictionary["config_parameters"]
        if model_class == VanillaTransformer:
            config_parameters["device"] = device

        # initializing the model to approximate
        model = model_class(model_dictionary["config_class"](**config_parameters))
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
        approx_model.to(device)
        # initializing the forward input
        forward_input = deepcopy(model_dictionary["forward_input"])
        for key, value in forward_input.items():
            if isinstance(value, Tensor):
                forward_input[key] = value.to(device)
        # forward pass
        output = approx_model(**forward_input)

        # ASSERTS

        # checking the model class
        assert isinstance(approx_model, nn.Module)
        assert isinstance(approx_model, model_class)
        # checking the output class
        assert isinstance(output, model_dictionary["output_class"])

        # releasing GPU memory, if needed
        if device == "cuda":
            torch.cuda.empty_cache()
