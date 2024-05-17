""" arrayfiller.py

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
from typing import ClassVar, Any

# Third-Party Packages #
from blockobjects import BaseBlock
import numpy as np

# Local Packages #


# Definitions #
# Classes #
class ArrayFiller(BaseBlock):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("a", "b")
    default_output_names: ClassVar[tuple[str, ...]] = ("a",)

    # Evaluate
    def evaluate(self, a: np.ndarray, b: np.ndarray, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        a[:] = b[:]
        return a
