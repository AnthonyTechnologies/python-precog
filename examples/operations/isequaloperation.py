""" isequaloperation.py
An Operation which checks if all values in the array are equal. Can select the equals methods.
"""
# Package Header #
from precog.header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from typing import Any

# Third-Party Packages #
from baseobjects.functions import MethodMultiplexer
import numpy as np

# Local Packages #
from precog.operations.operation import BaseOperation


# Definitions #
# Classes #
class IsEqualOperation(BaseOperation):
    """An Operation which checks if all values in the array are equal. Can select the equals methods.

    Attributes:
        inputs: The inputs manager of the Operation.
        outputs: The outputs manager of the Operation.
        execute: The method multiplexer which manages which execute method to run when called.
        input_names: The ordered tuple with the names of the inputs to an Operation.
        _output_names: The ordered tuple with the names of the outputs to an Operation.

        is_equal: The method multiplexer which manages which is_equal method to run when called.

    Args:
        equals_method: The name of the equals method to use.
        *args: Arguments for inheritance.
        init_io: Determines if construct_io run during this construction.
        setup: Determines if setup will run during this construction.
        init: Determines if this object will construct.
        **kwargs: Keyword arguments for inheritance.
    """
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names: ClassVar[tuple[str, ...]] = ("result",)
    default_equals_method: str = "all"

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        equals_method: str | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.is_equal: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_equals_method)

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(*args, init_io=init_io, equals_method=equals_method, sets_up=sets_up, **kwargs)

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        equals_method: str | None = None,
        *args: str | None, init_io: Any = True,
        sets_up: bool = True,
        setup_kwargs: bool = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            *args: Arguments for inheritance.
            sets_up: Determines if construct_io run during this construction.
            **kwargs: Keyword arguments for inheritance.
        """
        if equals_method is not None:
            self.is_equal.select(equals_method)

        # Construct Parent #
        super().construct(*args, init_io=sets_up, sets_up=sets_up, **kwargs)

    # Is Equal
    def all(self, data: np.ndarray) -> bool:
        """Checks if all the values in data are the same.

        Args:
            data: The array to check.

        Returns:
            The boolean if all values are the same.
        """
        return np.all(data[0])

    def unique(self, data: np.ndarray) -> bool:
        """Checks if all the values in data are unique.

        Args:
            data: The array to check.

        Returns:
            The boolean if all values are unique.
        """
        return np.unique(data).size == data.size

    # Evaluate
    def evaluate(self, data: np.ndarray, *args, **kwargs: Any) -> Any:
        """Checks if all values in the array are equal.

        Args:
            data: The array to check if all values are equal.
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The boolean if all values are equal.
        """
        return self.is_equal(data)
