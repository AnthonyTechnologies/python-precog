""" unitsumnomralization.py

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
from blockobjects import BaseBlock
import numpy as np

# Local Packages #


# Definitions #
# Classes #
class UnitSumNormalization(BaseBlock):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names: ClassVar[tuple[str, ...]] = ("n_data",)

    # Attributes #
    axis: int | tuple[int, int] | None = 0
    keep_dims: bool = True

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        keep_dims: bool | None = None,
        axis: int | tuple[int, int] | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                keep_dims=keep_dims,
                axis=axis,
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
        keep_dims: bool | None = None,
        axis: int | tuple[int, int] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            *args: Arguments for inheritance.
            **kwargs: Keyword arguments for inheritance.
        """
        if axis is not None:
            self.axis = axis

        if keep_dims is not None:
            self.keep_dims = keep_dims

        # Construct Parent #
        super().construct(*args, **kwargs)

    # Setup
    def setup(self, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        pass

    # Evaluate
    def evaluate(self, data: np.ndarray, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        return data / data.sum(axis=self.axis, keepdims=self.keep_dims)
