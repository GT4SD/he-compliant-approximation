""""Model approximation controller."""

import logging
import os
import re
from copy import deepcopy
from typing import Callable, Dict, Optional, Union

from torch import nn

from .aliases import ALIASES_FILE, load_modules_aliases

# all the approximator classes must be imported to let the controller know their existance
from .approximators.activation.quadratic import QuadraticApproximator  # noqa
from .approximators.activation.trainable_quadratic import (  # noqa
    TrainableQuadraticApproximator,
)
from .approximators.attention.masking.multiplicative import (  # noqa
    MultiplicativeAttentionMaskingApproximator,
)
from .approximators.attention.query_key_product.not_scaled import (  # noqa
    NotScaledQueryKeyDotProductApproximator,
)
from .approximators.core import ModuleApproximator
from .approximators.layer_normalization.batch_normalization import (  # noqa
    LayerNormToBatchNormApproximator,
)
from .approximators.layer_normalization.distill_layernorm import (  # noqa
    DistillLayerNormApproximator,
)
from .approximators.multihead.customizable_multihead import (  # noqa
    CustomizableMultiHeadApproximator,
)
from .approximators.softmax.mlp_softmax import MLPSoftmaxApproximator  # noqa
from .approximators.softmax.polynomial import PolynomialSoftmaxApproximator  # noqa
from .approximators.softmax.taylor import TaylorSoftmaxApproximator  # noqa
from .module_to_approximate import ToApproximate

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ModelApproximationController:
    """ModelApproximationController implementation."""

    def __init__(
        self,
        model: nn.Module,
        to_approximate: Optional[ToApproximate] = None,
        modules_aliases_file: str = ALIASES_FILE,
        save_path: str = os.getcwd(),
    ) -> None:
        """Initializes the ModelApproximationController.

        Args:
            model: model whose modules must be approximated.
            to_approximate: set of modules to be approximated. Defaults to None.
            modules_aliases_file: path of the file containing the modules' aliases informations. Defaults to ALIASES_FILE.
            save_path: saving directory path. Defaults to `os.getcwd()`.

        Raises:
            ValueError: in case of not existing model.
        """
        super().__init__()

        if model is None:
            raise ValueError("The model to be approximated must exist.")

        self.model = model
        # loading the modules' aliases mapping
        self.modules_aliases = load_modules_aliases(file_path=modules_aliases_file)

        self.save_path = save_path

        self.to_approximate: ToApproximate
        self.approximators: Dict[str, ModuleApproximator]
        if to_approximate is None:
            self.to_approximate = ToApproximate(**{"modules_set": set()})
            self.approximators = {}
        else:
            self.update_to_approximate(to_approximate=to_approximate)

    def update_model(self, model: nn.Module) -> None:
        """Updates the model to be approximated by the controller.

        Args:
            model: model whose modules must be approximated..
        """
        self.model = model
        self.approximated_model = deepcopy(self.model)
        self.is_approximated = False
        self.is_pretrained = False

    def update_save_path(self, save_path: str) -> None:
        """Updates the saving path.

        Args:
            save_path: saving directory path.
        """
        self.save_path = save_path

    def update_to_approximate(
        self, to_approximate: ToApproximate, verbose: bool = True
    ) -> None:
        """Updates the set of modules to be approximated, collecting the approximators.

        Args:
            to_approximate: set of modules to be approximated.
            verbose: whether to print some informations in stdout. Defaults to True.

        Raises:
            ValueError: in case of empty list of modules to be approximated.
            ValueError: in case of empty set of approximators for the list of modules to be approximated.
        """
        self.to_approximate = to_approximate

        if self.to_approximate is None:
            raise ValueError(
                "The list of module to replace must contain some module's class."
            )

        # RECALL: to be able to gather all ModuleApproximator subclasses they must be imported
        approximators_classes = ModuleApproximator.__subclasses__()

        self.approximated_model = deepcopy(self.model)
        self.is_approximated = False
        self.is_pretrained = False

        # creating the dictionary that assigns the requested approximator to the module type
        self.approximators = {}
        exists_approximator = False
        # assinging the approximator for each ModuleToApproximate
        for module_to_approximate in self.to_approximate.get_modules_set():
            exists_approximator = False
            aliases = self.modules_aliases.get_module_aliases(
                module_name=module_to_approximate.module
            )
            if module_to_approximate.approximation_type == "":
                approximation_type = (
                    self.modules_aliases.get_module_default_approximation_type(
                        module_to_approximate.module
                    )
                )
            else:
                approximation_type = module_to_approximate.approximation_type
            for approximator in approximators_classes:
                supported_layer_types = set(
                    [
                        ModelApproximationController.get_object_module(module)
                        for module in approximator.supported_layer_types
                    ]
                )
                if (
                    aliases.issubset(supported_layer_types)
                    and approximation_type == approximator.approximation_type
                ):
                    exists_approximator = True
                    common_approximator = approximator(
                        parameters=module_to_approximate.parameters,
                    )
                    for alias in aliases:
                        # the module is associated to its approximator instance
                        self.approximators[alias] = common_approximator
                    break

            if not exists_approximator and verbose:
                print(
                    f"No approximator available for {{{module_to_approximate}}}. Inherit ModuleApproximator to implement an approximation for module= {{{module_to_approximate}}}."
                )
            exists_approximator = False
            dependencies = self.modules_aliases.get_module_dependencies(
                module_name=module_to_approximate.module
            )
            if not dependencies == []:
                for dep in dependencies:
                    if dep.module not in self.approximators.keys():
                        aliases = self.modules_aliases.get_module_aliases(
                            module_name=dep.module
                        )
                        if dep.approximation_type == "":
                            approximation_type = self.modules_aliases.get_module_default_approximation_type(
                                dep.module
                            )
                        else:
                            approximation_type = dep.approximation_type
                        for approximator in approximators_classes:
                            supported_layer_types = set(
                                [
                                    ModelApproximationController.get_object_module(
                                        module
                                    )
                                    for module in approximator.supported_layer_types
                                ]
                            )
                            if (
                                aliases.issubset(supported_layer_types)
                                and approximation_type
                                == approximator.approximation_type
                            ):
                                exists_approximator = True
                                common_approximator = approximator(
                                    parameters=dep.parameters,
                                )
                                for alias in aliases:
                                    # the module type is associated to its approximator instance
                                    self.approximators[alias] = common_approximator
                                break

                        if not exists_approximator and verbose:
                            print(
                                f"No approximator available for {{{dep}}}. Inherit ModuleApproximator to implement an approximation for module= {{{dep}}}."
                            )

        if self.approximators == {}:
            raise ValueError(
                f"The set of approximators is empty. Implement approximators for {{{self.to_approximate}}}."
            )

    @staticmethod
    def get_object_module(object: Union[nn.Module, Callable]) -> str:
        """Gets the string that identifies the module of the object.

        Args:
            object: module/function to identify.

        Raises:
            AttributeError: in case the object has not __module__ attribute.

        Returns:
            string identifing the module of the object
        """
        if hasattr(object, "__module__") and hasattr(object, "__name__"):
            return f"{object.__module__}.{object.__name__}"
        elif hasattr(object, "__module__"):
            # defining a regular-expression to select the name from the class i.e. the string after the last '.' character.
            regex = r"(?<=[ .])(\w+)'>"
            # retrieving the name using the latter regular-expression
            class_name = re.search(regex, str(object.__class__)).group(1)  # type: ignore
            return f"{object.__module__}.{class_name}"  # type: ignore
        else:
            raise AttributeError(
                f"Impossible to get the module name of the object {object}."
            )

    def print_model_structure(self) -> None:
        """Prints the model structure."""
        print(f"{self.model}")

    def print_approximated_model_structure(self) -> None:
        """Prints the appriximated model structure."""
        if not self.is_approximated:
            print(
                "Run the 'get_approximated_model()' controller's method to approximate the model."
            )
        else:
            print(f"{self.approximated_model}")

    def print_available_approximators(self) -> None:
        """Prints the pairs of module type and its approximator for the module."""
        for id, (key, value) in enumerate(self.approximators.items()):
            print(f"({id+1})  {key} approximated by {value}")

    def recursive_search_with_approximation(
        self, model: Optional[nn.Module] = None, pretrained: bool = False
    ) -> Dict[int, int]:
        """Implements the recursive search inside the model, applying the approximation.

        Args:
            model: module to inspect recursively. Defaults to None.
            pretrained: whether to return the approximated model with pretrained or trainable layers. Defaults to False.

        Returns:
            dictionary containing the number of substituted layers for each available approximator.
        """

        num_affected = dict(
            zip(
                range(1, len(self.approximators.keys()) + 1),
                [0] * len(self.approximators.keys()),
            )
        )

        # handling the model selection for the first call of the function
        if model is None:
            model = self.approximated_model

        # iterating through the layers of the network
        for id, module in model.named_children():
            # in case it is a simple module
            if (
                ModelApproximationController.get_object_module(module)
                in list(self.approximators.keys())
                and not pretrained
            ) or (pretrained and getattr(module, "is_trainable", False)):
                # if we have to approximate the model ready for a possible training
                if (
                    ModelApproximationController.get_object_module(module)
                    in list(self.approximators.keys())
                    and not pretrained
                ):
                    string = ModelApproximationController.get_object_module(module)
                elif pretrained and getattr(module, "is_trainable", False):
                    if getattr(module, "is_approximation_of", None) is None:
                        raise ValueError(
                            f"{module.__class__} should have an attribute 'is_approximation_of'."
                        )
                    string = ModelApproximationController.get_object_module(
                        getattr(module, "is_approximation_of")
                    )
                layer_index = list(self.approximators.keys()).index(string) + 1

                # approximating the layer
                setattr(
                    model,
                    id,
                    self.approximators[f"{string}"].approximate_module(
                        model=model,
                        id=id,
                        pretrained=pretrained,
                        save_path=self.save_path,  # type: ignore
                    ),
                )
                # updating the value to the substituted layer, to keep track of the changes done
                module = getattr(model, id)
                # updating the number of approximated layers
                num_affected[layer_index] = num_affected[layer_index] + 1

            # checking the arguments of the module for possible subsitutions
            attributes_to_change = {}
            for key, value in vars(module).items():
                if key.startswith("_") or not callable(value):
                    continue
                if (
                    ModelApproximationController.get_object_module(value)
                    in [key for key in self.approximators.keys()]
                    and not pretrained
                ):
                    attributes_to_change[key] = value

            # NOTE: the change is NOT done in the latter loop since we would obtain an error
            # indeed the dictionary would change while looping on it
            for key, value in attributes_to_change.items():
                string = ModelApproximationController.get_object_module(value)
                # approximating the layer
                setattr(
                    module,
                    key,
                    self.approximators[f"{string}"].approximate_module(
                        model=module,
                        id=key,
                        pretrained=pretrained,
                        save_path=self.save_path,  # type: ignore
                    ),
                )
                layer_index = list(self.approximators.keys()).index(string) + 1
                num_affected[layer_index] = num_affected[layer_index] + 1

            # in case of a compound module, go inside it
            # (this must be done after the possible approximation of the module done above)
            if len(list(module.children())) > 0 and getattr(
                module, "allow_recursive_search", True
            ):
                to_add = self.recursive_search_with_approximation(
                    model=module, pretrained=pretrained
                )
                num_affected = {
                    k: num_affected.get(k, 0) + to_add.get(k, 0)
                    for k in set(num_affected) & set(to_add)
                }

        return num_affected

    def get_approximated_model(
        self, pretrained: bool = False, verbose: bool = False
    ) -> nn.Module:
        """Replaces the model modules with the approximated version.

        Args:
            verbose: whether to log the informations about the subsitution. Defaults to False.

        Returns:
            approximated model.
        """

        if self.to_approximate.get_modules_set() == set():
            print(
                "Specify which modules shoud be approximated. Use 'update_to_approximate' class method."
            )

        if self.is_approximated and self.is_pretrained:
            if verbose:
                print("Model was already approximated in its pretrained form.")
            return self.approximated_model

        elif self.is_approximated and not pretrained:
            if verbose:
                print("Model was already approximated in its trainable form.")
            return self.approximated_model

        elif not self.is_approximated and pretrained:
            print(
                "Model must be approximated in its trainable form before being converted to its pretrained form."
            )
            return self.model

        if self.is_approximated and not self.is_pretrained and pretrained and verbose:
            print(
                "Model will be be transformed from its trainable to its pretrained form."
            )
        elif not self.is_approximated and pretrained and verbose:
            print(
                "Model will be be approximated from its original to its trainable form."
            )

        if verbose:
            # logging available approximators
            print("The following types of layers are going to be substituted:")
            self.print_available_approximators()

        # approximate the model
        affected = self.recursive_search_with_approximation(pretrained=pretrained)
        self.is_approximated = True
        self.is_pretrained = pretrained

        if verbose:
            # logging how many layers were affected by the approximation
            print(
                f"A total of {sum(affected.values())} of layers were substituted with an approximation approximated. In particular:"
            )
            for key, value in affected.items():
                print(f" - For ({key}) a total of {value} approximated layers.")

        return self.approximated_model
