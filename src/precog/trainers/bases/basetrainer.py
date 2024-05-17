""" basetrainer.py.py
An abstract base class for model trainers.
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
class BaseTrainer(BasisContainer):
    """An abstract base class for model trainers.

    Class Attributes:
        default_subtrainers: Contains the default subtrainer types and their keyword arguments.

    Attributes:
        local_state_variables: The local state variables of this trainer object.
        subtrainers: Contains the names of the subtrainers and their corresponding subtrainer objects.

    Args:
        bases: The model bases related to this trainer.
        state_variables: The local state variables to assign to this trainer.
        subtrainers: The subtrainers to assign to this trainer.
        *args: The positional arguments for this trainer.
        create_defaults: Determines if the default state variables and subtrainers will be created.
        bases_kwargs: The keyword arguments for the default bases.
        subtrainers_kwargs: The keyword arguments for the default subtrainers.
        init: Determines if this trainer will initialize.
        **kwargs: The keyword arguments for this trainer.
    """
    # Class Attributes #
    default_subtrainers: ClassVar[dict[str, tuple[type[ModelBasis], dict[str, Any]]]] = {}

    # Attributes #
    local_state_variables: dict[str, Any]
    subtrainers: dict[str, "BaseTrainer"]

    # Properties #
    @property
    def state_variables(self) -> dict[str, Any]:
        """The state variables of this trainer object."""
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
        subtrainers: dict[str, "BaseTrainer"] | None = None,
        *args: Any,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        subtrainers_kwargs: dict[str, dict[str, Any]] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Attributes #
        self.local_state_variables = {}
        self.subtrainers = {}

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                bases=bases,
                state_variables=state_variables,
                subtrainers=subtrainers,
                create_defaults=create_defaults,
                bases_kwargs=bases_kwargs,
                subtrainers_kwargs=subtrainers_kwargs,
                **kwargs,
            )

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        subtrainers: dict[str, "BaseTrainer"] | None = None,
        *args: Any,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        subtrainers_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            bases: The model bases related to this trainer.
            state_variables: The local state variables to assign to this trainer.
            subtrainers: The subtrainers to assign to this trainer.
            *args: The positional arguments for this trainer.
            create_defaults: Determines if the default state variables and subtrainers will be created.
            bases_kwargs: The keyword arguments for the default bases.
            subtrainers_kwargs: The keyword arguments for the default subtrainers.
            **kwargs: The keyword arguments for this trainer.
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
            self.construct_default_subtrainers(subtrainers_kwargs=subtrainers_kwargs)
        if subtrainers is not None:
            self.subtrainers.update(subtrainers)

    # State Variables
    def get_state_variables(self) -> dict[str, Any]:
        """Gets the state variables of this trainer.

        Returns:
            A dictionary containing:
            "local": The state variables of this instance of the trainer.
            "subtrainers": A dictionary containing the names of the subtrainers and their corresponding state variables.
        """
        return {
            "local": self.local_state_variables,
            "subtrainers": {n: s.state_variables for n, s in self.subtrainers.items()},
        }

    # Trainers
    def construct_default_subtrainers(self, subtrainers_kwargs: dict[str, dict[str, Any]] | None = None) -> None:
        """Constructs the default subtrainers for this trainer.

        Args:
            subtrainers_kwargs: The keyword arguments for the default subtrainers.
        """
        if subtrainers_kwargs is None:
            self.subtrainers.update({n: t(**k) for n, (t, k) in self.default_subtrainers.items()})
        else:
            self.subtrainers.update(
                {n: t(**(k | subtrainers_kwargs.get(n, {}))) for n, (t, k) in self.default_subtrainers.items()},
            )
