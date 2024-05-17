""" basefeature.py

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
from abc import abstractmethod
from typing import ClassVar, Any

# Third-Party Packages #
from blockobjects import BaseBlock
import numpy as np

# Local Packages #


# Definitions #
# Classes #
class BaseFeature(BaseBlock):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names: ClassVar[tuple[str, ...]] = ("features",)

    # Instance Methods #
    # Evaluate
    @abstractmethod
    def evaluate(self, data: np.ndarray, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            data: The array to create features from.
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        pass

