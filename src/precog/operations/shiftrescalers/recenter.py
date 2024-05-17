""" recenter.py

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
from baseobjects import MethodMultiplexer
from blockobjects import BaseBlock
import numpy as np

# Local Packages #


# Definitions #
# Classes #
class Recenter(BaseBlock):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names: ClassVar[tuple[str, ...]] = ("r_data",)

    default_find_center: ClassVar[str] = "find_peak"
    default_recenter: ClassVar[str] = "roll"

    # New Attributes #
    axis: int | tuple[int, int] | None = 0

    find_center: MethodMultiplexer
    recenter: MethodMultiplexer

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        axis: int | tuple[int, int] | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.find_center = MethodMultiplexer(instance=self, select=self.default_find_center)
        self.recenter = MethodMultiplexer(instance=self, select=self.default_recenter)

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                axis=axis,
                *args,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
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

        # Construct Parent #
        super().construct(*args, **kwargs)

    # Setup
    def setup(self, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        pass

    # Find Center
    def find_peak(self, data: np.ndarray):
        return data.argmax()

    def find_center_mass(self, data: np.ndarray):
        means = np.moveaxis(data, self.axis, 0).mean(axis=-1)
        center = ((means / means.sum()) * range(len(means))).sum()
        return 0 if np.isnan(center) else int(center)

    # Recenter
    def roll(self, data: np.ndarray):
        return np.roll(data, data.shape[self.axis] // 2 - self.find_center(), axis=self.axis)

    def reflect_roll(self, data: np.ndarray):
        slices = [slice(s) for s in data.shape]
        axis_len = data.shape[self.axis]
        shift = axis_len // 2 - self.find_center()
        width = [(0, 0)] * len(data.shape)
        width[self.axis] = (shift, 0) if shift > 0 else (0, -shift)
        if shift > 0:
            width[self.axis] = (shift, 0)
        else:
            width[self.axis] = (0, -shift)
            slices[self.axis] = slice(1 - axis_len)

        return np.pad(data, width, mode="reflect")[tuple(slices)]

    # Evaluate
    def evaluate(self, data: np.ndarray, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        return self.recenter(data)
