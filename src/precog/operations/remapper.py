""" remapper.py

"""
# Package Header #
from ..header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from copy import deepcopy
from typing import ClassVar, Any

# Third-Party Packages #
from blockobjects import BaseBlock
import numpy as np

# Local Packages #


# Definitions #
# Classes #
class Remapper(BaseBlock):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data", "map_matrix")
    default_required_input: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names: ClassVar[tuple[str, ...]] = ("remapped_data",)

    # New Attributes #
    axis: int = 1
    map_matrix: np.ndarray | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        map_matrix: np.ndarray | None = None,
        axis: int | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                *args,
                map_matrix=map_matrix,
                axis=axis,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        map_matrix: np.ndarray | None = None,
        axis: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            map_matrix: The remap matrix to apply.
            axis: The axis to remap along.
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            sets_up: Determines if setup will run during this construction.
            setup_kwargs: The keyword arguments for the setup method.
            **kwargs: Keyword arguments for inheritance.
        """
        if map_matrix is not None:
            self.map_matrix = map_matrix

        if axis is not None:
            self.axis = axis

        # Construct Parent #
        super().construct(*args, **kwargs)

    # Setup
    def setup(self, map_matrix: np.ndarray | None = None, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        if map_matrix is not None:
            self.map_matrix = map_matrix

    # Evaluate
    def evaluate(self, data: np.ndarray, map_matrix: np.ndarray | None = None, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            data: The array to remap.
            map_matrix: The remap matrix to apply.
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        # Input
        if map_matrix is not None:
            self.map_matrix = map_matrix

        # Remap
        remapped = np.moveaxis(np.moveaxis(data, self.axis, -1) @ self.map_matrix, -1, self.axis)

        # Output
        return remapped
