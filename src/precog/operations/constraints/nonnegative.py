""" nonnegative.py

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
from copy import deepcopy
from typing import ClassVar, Any

# Third-Party Packages #
from baseobjects.functions import CallableMultiplexObject, MethodMultiplexer
from blockobjects import BaseBlock
import numpy as np

# Local Packages #


# Definitions #
# Classes #
class NonNegative(BaseBlock):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names: ClassVar[tuple[str, ...]] = ("nn_data",)

    default_non_negative: ClassVar[str] = "clip"

    # Attributes #
    non_negative: MethodMultiplexer
    non_negative_kwargs: dict = {}

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        non_negative: str | None = None,
        non_negative_kwargs: dict[str, Any] | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.non_negative: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_non_negative)
        self.non_negative_kwargs: dict = self.non_negative_kwargs.copy()

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                *args,
                non_negative=non_negative,
                non_negative_kwargs=non_negative_kwargs,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        non_negative: str | None = None,
        non_negative_kwargs: dict[str, Any] | None = None,
        *args: str | None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            *args: Arguments for inheritance.
            **kwargs: Keyword arguments for inheritance.
        """
        if non_negative is not None:
            self.non_negative.select(non_negative)

        if non_negative_kwargs is not None:
            self.non_negative_kwargs.clear()
            self.non_negative_kwargs.update(non_negative_kwargs)

        # Construct Parent #
        super().construct(*args, **kwargs)

    # Non-Negative
    def clip(self, data: np.ndarray, threshold: float = 0, **kwargs: Any) -> np.ndarray:
        return data.clip(min=threshold)  # threshold >= 0

    def abs(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.abs(data)

    def square(self, data: np.ndarray, **kwargs: Any) -> np.ndarray:
        return data**2

    # Evaluate
    def evaluate(self, data: np.ndarray, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        nn_data = self.non_negative(data, **self.non_negative_kwargs)

        # Output
        return nn_data
