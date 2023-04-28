"""Module approximation data structures."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from pydantic import BaseModel, conlist, root_validator


class ModuleToApproximate(BaseModel):
    """Contains the information for the module approximation.

    Attributes:
        module: name of the module to approximate.
        approximation_type: name of the type of approximation.
        parameters: parameters of the approximation module.
    """

    module: str
    approximation_type: str
    parameters: Dict[str, Any]

    @root_validator
    def check_batch_normalization_with_TID_parameters(
        cls: ModuleToApproximate, values: Any
    ) -> Any:
        """Validates the parameters related to the Regularized BatchNorm approximation.

        Args:
            cls: module to approximate.
            values: values of the module to approximate.

        Raises:
            ValueError: with a Regularized BatchNorm the loss hyperparameters 'alpha' and 'beta' must be specified.

        Returns:
            values of the module to approximate.
        """
        approximation_type, parameters = values.get("approximation_type"), values.get(
            "parameters"
        )
        if (
            approximation_type == "batchnorm"
            and parameters.get("regularized_BN", None) is not None
        ):
            # checking whether the batch normalization parameters are set correctly
            if (
                parameters.get("RBN_alpha") is None
                and parameters.get("RBN_beta") is None
            ):
                raise ValueError(
                    "Regularized Batch Normalization should run setting 'RBN_alpha' and 'RBN_beta' values."
                )
            elif parameters.get("RBN_alpha") is None:
                raise ValueError(
                    "Regularized Batch Normalization should run setting 'RBN_alpha' value."
                )
            elif parameters.get("RBN_beta") is None:
                raise ValueError(
                    "Regularized Batch Normalization should run setting 'RBN_beta' value."
                )
        return values

    def __hash__(self) -> int:
        """Fetches the hash value of the module to approximate object.

        Returns:
            hash value of the module to approximate object.
        """
        return hash(self.module)


class ToApproximate(BaseModel):
    """Contains the set of modules to approximate.

    Attributes:
        modules_set: set of modules to approximate.
        _modules_name_list: list of the modules' names
    """

    modules_set: conlist(item_type=ModuleToApproximate, unique_items=True)  # type: ignore

    # declaring the attributes to be defined during initialization
    __slots__ = ["_modules_name_list"]

    def __init__(self, **data: Any) -> None:
        """Initializes the data structure."""
        super().__init__(**data)
        # defining the list of the modules' names
        object.__setattr__(
            self,
            "_modules_name_list",
            [item.module for item in self.modules_set],
        )

    def get_module_name_list(self) -> List[ModuleToApproximate]:
        """Gets the list of the modules' names

        Returns:
            list of the modules' names
        """
        return self._modules_name_list  # type: ignore

    def is_module_in_set(self, module_name: str) -> bool:
        """Checks the existence of a module name in the list of the modules' names.

        Args:
            module_name: name of the module to check the existence of.

        Returns:
            whether the module name is in the list of the modules' names or not.
        """
        if module_name in self._modules_name_list:  # type: ignore
            return True
        else:
            return False

    def get_modules_set(self) -> Set[ModuleToApproximate]:
        """Gets the set of modules to approximate.

        Returns:
            set of modules to approximate.
        """
        return set(self.modules_set)

    def set_modules_set(self, new_modules_set: Set) -> None:
        """Updates the set of modules to approximate.

        Args:
            new_modules_set: new set of modules to approximate.
        """
        self.modules_set = new_modules_set
        object.__setattr__(
            self,
            "_modules_name_list",
            [item.module for item in self.modules_set],
        )
