"""Modules' aliases data structure."""

from __future__ import annotations

import json
import os
from typing import Any, List, Set

import importlib_resources
from pydantic import BaseModel, conlist

from .module_to_approximate import ModuleToApproximate

ALIASES_FILE = str(
    importlib_resources.files("hela") / "resources" / "approximation" / "aliases.json"
)


class ModuleAliases(BaseModel):
    """Contains the information for the mapping between the common module name and the torch module.

    Attributes:
        name: common name to easily identify the module.
        aliases: set of classes' names to be mapped with the common module name.
        default_approximation_type: name of the approximation type to use if not specified.
        dependencies: additional set of modules to be approximated, even if not specified, due to an approximation dependency.
    """

    name: str
    aliases: Set[str]
    default_approximation_type: str
    dependencies: conlist(item_type=ModuleToApproximate, unique_items=True)  # type: ignore

    def __hash__(self) -> int:
        """Fetches the hash value of the module aliases object.

        Returns:
            hash value of the module aliases object.
        """
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Compares two module aliases objects using their name attribute.

        Args:
            other: module aliases to be compared to.

        Returns:
            whether the two module aliases are referred to the same module.
        """
        if not isinstance(other, ModuleAliases):
            return NotImplemented
        return self.name == other.name


class Aliases(BaseModel):
    """Contains the list of all the modules' aliases.

    Attributes:
        aliases_list: list of all the modules' aliases.
        _aliases_dict: dictionary mapping the module name with its aliases.
        _dependencies_dict: dictionary mapping the module name with its dependencies.
        _default_approximation_type_dict: dictionary mapping the module name with its default approximation type.
    """

    aliases_list: conlist(item_type=ModuleAliases, unique_items=True)  # type: ignore

    # declaring the attributes to be defined during initialization
    __slots__ = [
        "_aliases_dict",
        "_dependencies_dict",
        "_default_approximation_type_dict",
    ]

    def __init__(self, **data: Any) -> None:
        """Initializes the data structure."""
        super().__init__(**data)
        # defining the name to module mapping dictionary
        object.__setattr__(
            self,
            "_aliases_dict",
            {
                module_alias.name: set(module_alias.aliases)
                for module_alias in self.aliases_list
            },
        )
        # defining the dependencies dictionary
        object.__setattr__(
            self,
            "_dependencies_dict",
            {
                module_alias.name: set(module_alias.dependencies)
                for module_alias in self.aliases_list
            },
        )
        # defining the default approximation type dictionary
        object.__setattr__(
            self,
            "_default_approximation_type_dict",
            {
                module_alias.name: module_alias.default_approximation_type
                for module_alias in self.aliases_list
            },
        )

    def get_aliases_list(self) -> List[ModuleAliases]:
        """Gets the list of all the modules' aliases.

        Returns:
            list of all the modules' aliases.
        """
        return self.aliases_list

    def get_module_aliases(self, module_name: str) -> Set[str]:
        """Gets the set of aliases associated to a module.

        Args:
            module_name: common name of the module.

        Returns:
            set of aliases associated to the module.
        """
        return self._aliases_dict.get(module_name, set())  # type: ignore

    def get_module_dependencies(self, module_name: str) -> Set[ModuleToApproximate]:
        """Gets the set of dependencies associated to a module.

        Args:
            module_name: common name of the module.

        Returns:
            set of dependencies associated to the module.
        """
        return self._dependencies_dict.get(module_name, set())  # type: ignore

    def get_module_default_approximation_type(self, module_name: str) -> str:
        """Gets the default approximation type associated to a module.

        Args:
            module_name: common name of the module.

        Returns:
            default approximation type associated to the module.
        """
        return self._default_approximation_type_dict.get(module_name, "")  # type: ignore


def load_modules_aliases(file_path: str = ALIASES_FILE) -> Aliases:
    """Loads the modules' aliases from a file.

    Args:
        file_path: path of the file to be parsed.

    Returns:
        modules' aliases data structure.
    """
    return Aliases.parse_file(file_path)


def save_modules_aliases(file_path: str, save_path: str) -> None:
    """Saves the modules' aliases data structure in a json file.

    Args:
        file_path: modules' aliases file path.
        save_path: path of the saving directory.
    """
    aliases = Aliases.parse_file(file_path)

    save_path = os.path.join(save_path, "aliases.json")
    with open(save_path, "w") as outfile:
        json.dump(json.loads(aliases.json()), outfile, indent=4)
