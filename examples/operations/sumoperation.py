""" sumoperation.py
An Operation which scales the input ndarray and returns its sum, minium, and maximum.
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
import numpy as np

# Local Packages #
from precog.operations.operation import BaseOperation


# Definitions #
# Classes #
class SumOperation(BaseOperation):
    """An Operation which scales the input ndarray and returns its sum, minium, and maximum."""
    default_input_names: ClassVar[tuple[str, ...]] = ("data", "scale")
    default_output_names: ClassVar[tuple[str, ...]] = ("out_number", "scaled_min", "scaled_max")

    # Evaluate
    def evaluate(self, data: np.ndarray, scale: float | None = None, *args, **kwargs: Any) -> Any:
        """Scales the given ndarray and returns some information from the array.

        Args:
            data: The ndarray to scale and return information on.
            scale: The amount to scale the data by.
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The sum, min, and max of the scaled array.
        """
        if scale is None:
            scale = 1

        scaled_data = data * scale

        return scaled_data.sum(), scaled_data.min(), scaled_data.max()
