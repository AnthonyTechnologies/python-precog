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
from torch import Tensor

# Local Packages #
from ..adaptivemultiplicativemodifier import AdaptiveMultiplicativeModifier
from .basismodiferblock import BasisModifierBlock
from ...torch import TorchModelBasis


# Definitions #
# Classes #
class AdaptiveMultiplicativeBlock(BasisModifierBlock):
    modifier_type: type[AdaptiveMultiplicativeModifier] = AdaptiveMultiplicativeModifier

    # Evaluate
    def evaluate(
        self,
        data: np.ndarray | None = None,
        bases: dict[str, TorchModelBasis] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        if bases is not None:
            for name, new_basis in bases.items():
                if (basis := self.modifier.all_bases.get(name, None)) is not None:
                    if isinstance(new_basis, TorchModelBasis):
                        new_basis = new_basis.tensor

                    if isinstance(new_basis, np.ndarray):
                        basis.ndarray[...] = new_basis[...]
                    elif isinstance(new_basis, Tensor):
                        if basis.tensor is not new_basis:
                            basis.tensor[...] = new_basis[...]
        return self.modifier.update(x=data, *args, **kwargs)
