""" basearchitecture.py.py
An abstract base class for model architectures.
"""
# Package Header #
from ...header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from typing import ClassVar, Any

# Third-Party Packages #

# Local Packages #
from ...basis import ModelBasis, BasisContainer


# Definitions #
# Classes #
class BaseArchitecture(BasisContainer):
    """An abstract base class for model architectures.

    Class Attributes:
        default_subarchitectures: Contains the default subarchitecture types and their keyword arguments.

    Attributes:
        local_state_variables: The local state variables of this architecture object.
        subarchitectures: Contains the names of the subarchitectures and their corresponding subarchitecture objects.

    Args:
        bases: The model bases related to this architecture.
        state_variables: The local state variables to assign to this architecture.
        subarchitectures: The subarchitectures to assign to this architecture.
        *args: The positional arguments for this architecture.
        create_defaults: Determines if the default state variables and subarchitectures will be created.
        bases_kwargs: The keyword arguments for the default bases.
        subarchitectures_kwargs: The keyword arguments for the default subarchitectures.
        init: Determines if this architecture will initialize.
        **kwargs: The keyword arguments for this architecture.
    """
    # Class Attributes #
    default_subarchitectures: ClassVar[dict[str, tuple[type[ModelBasis], dict[str, Any]]]] = {}

    # Attributes #
    local_state_variables: dict[str, Any]
    subarchitectures: dict[str, "BaseArchitecture"]

    # Properties #
    @property
    def state_variables(self) -> dict[str, Any]:
        """The state variables of this architecture object."""
        return self.get_state_variables()

    @state_variables.setter
    def state_variables(self, value: dict[str, Any]) -> None:
        self.local_state_variables = value

    # Instance Methods #
    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        subarchitectures: dict[str, "BaseArchitecture"] | None = None,
        *args: Any,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        subarchitectures_kwargs: dict[str, dict[str, Any]] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Attributes #
        self.local_state_variables = {}
        self.subarchitectures = {}

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                bases=bases,
                state_variables=state_variables,
                subarchitectures=subarchitectures,
                create_defaults=create_defaults,
                bases_kwargs=bases_kwargs,
                subarchitectures_kwargs=subarchitectures_kwargs,
                **kwargs,
            )

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        subarchitectures: dict[str, "BaseArchitecture"] | None = None,
        *args: Any,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        subarchitectures_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.
        
        Args:
            bases: The model bases related to this architecture.
            state_variables: The local state variables to assign to this architecture.
            subarchitectures: The subarchitectures to assign to this architecture.
            *args: The positional arguments for this architecture.
            create_defaults: Determines if the default state variables and subarchitectures will be created.
            bases_kwargs: The keyword arguments for the default bases.
            subarchitectures_kwargs: The keyword arguments for the default subarchitectures.
            **kwargs: The keyword arguments for this architecture.
        """
        # Construct Parent #
        super().construct(
            bases=bases,
            state_variables=state_variables,
            create_defaults=create_defaults,
            bases_kwargs=bases_kwargs,
            **kwargs,
        )

        # Construct New #
        if create_defaults:
            self.construct_default_subarchitectures(subarchitectures_kwargs=subarchitectures_kwargs)
        if subarchitectures is not None:
            self.subarchitectures.update(subarchitectures)

    # State Variables
    def get_state_variables(self) -> dict[str, Any]:
        """Gets the state variables of this architecture.

        Returns:
            A dictionary containing:
            "local": The state variables of this instance of the architecture.
            "subarchitectures": A dictionary containing the names of the subarchitectures and their corresponding state variables.
        """
        return {
            "local": self.local_state_variables,
            "subarchitectures": {n: s.state_variables for n, s in self.subarchitectures.items()},
        }

    # Architectures
    def construct_default_subarchitectures(self, subarchitectures_kwargs: dict[str, dict[str, Any]] | None = None) -> None:
        """Constructs the default subarchitectures for this architecture.

        Args:
            subarchitectures_kwargs: The keyword arguments for the default subarchitectures.
        """
        if subarchitectures_kwargs is None:
            self.subarchitectures.update({n: t(**k) for n, (t, k) in self.default_subarchitectures.items()})
        else:
            self.subarchitectures.update({
                n: t(**(k | subarchitectures_kwargs.get(n, {}))) for n, (t, k) in self.default_subarchitectures.items()
            })
