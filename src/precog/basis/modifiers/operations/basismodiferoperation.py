""" basismodifieroperation.py.py

"""
# Package Header #
from ....header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__

# Imports #
# Standard Libraries #
from typing import ClassVar, Any

# Third-Party Packages #
from blockobjects import BaseBlock
import numpy as np

# Local Packages #
from ..basebasismodifier import BaseBasisModifier


# Definitions #
# Classes #
class BasisModifierOperation(BaseBlock):
    default_input_names: ClassVar[tuple[str, ...]] = ("data", "bases")
    default_output_names: ClassVar[tuple[str, ...]] = ("m_data",)
    modifier_type: type[BaseBasisModifier] | None = None

    # New Attributes #
    modifier: BaseBasisModifier | None = None

    # Properties #
    @property
    def state_variables(self) -> dict[str, Any]:
        return self.modifier.state_variables

    @state_variables.setter
    def state_variables(self, value: dict[str, Any]) -> None:
        self.modifier.state_variables = value

    @property
    def bases(self) -> dict[str, Any]:
        return self.modifier.bases

    @bases.setter
    def bases(self, value: dict[str, Any]) -> None:
        self.modifier.bases = value

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        modifier: BaseBasisModifier | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                modifier=modifier,
                *args,
                init_io=init_io,
                sets_up=sets_up,
                setup_kwargs=setup_kwargs,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        modifier: BaseBasisModifier | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            sets_up: Determines if setup will run during this construction.
            setup_kwargs: The keyword arguments for the setup method.
            **kwargs: Keyword arguments for inheritance.
        """
        if modifier is not None:
            self.modifier = modifier

        # Construct Parent #
        super().construct(*args, init_io=init_io, sets_up=sets_up, setup_kwargs=setup_kwargs, **kwargs)

    # Setup
    def setup(self, modifier_kwargs: dict[str, Any] = None, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        if self.modifier is None:
            self.modifier = self.modifier_type(**({} if modifier_kwargs is None else modifier_kwargs))

    # Evaluate
    def evaluate(self, data: np.ndarray | None = None, bases: None = None, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        return self.modifier.update(data, *args, **kwargs)