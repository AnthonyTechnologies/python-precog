""" adaptivemultiplicativeblock.py

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
from typing import Any

# Third-Party Packages #
import numpy as np

# Local Packages #
from ..adaptivemultiplicativemodifier import AdaptiveMultiplicativeModifier
from .basismodiferblock import BasisModifierBlock


# Definitions #
# Classes #
class AdaptiveMultiplicativeBlock(BasisModifierBlock):
    modifier_type: type[AdaptiveMultiplicativeModifier] = AdaptiveMultiplicativeModifier

    # Evaluate
    def evaluate(self, data: np.ndarray | None = None, bases: dict | None = None, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        return None if data is None else self.modifier.update(x=data, *args, **kwargs)
