""" linelength.py

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
from typing import ClassVar, Any, Callable

# Third-Party Packages #
import numpy as np
from blockobjects import BaseBlock
from proxyarrays import BaseProxyArray
from scipy.signal import convolve
from scipy.signal.windows import hann

# Local Packages #
from .basefeature import BaseFeature


# Definitions #
# Classes #
class LineLength(BaseFeature):

    # Attributes #
    axis: int = 0
    squared_estimator: bool = False
    window_len: int = 0
    window_type: Callable = hann

    preappend: np.ndarray | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        squared_estimator: bool | None = None,
        window_len: int | None = None,
        window_type: Callable | None = None,
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
                squared_estimator=squared_estimator,
                window_len=window_len,
                window_type=window_type,
                axis=axis,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        squared_estimator: bool | None = None,
        window_len: int | None = None,
        window_type: Callable | None = None,
        axis: int | None = None,
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

        if squared_estimator is not None:
            self.squared_estimator = squared_estimator

        if window_len is not None:
            self.window_len = window_len

        if window_type is not None:
            self.window_type = window_type

        # Construct Parent #
        super().construct(*args, **kwargs)

    # Evaluate
    def evaluate(self, data: np.ndarray | BaseProxyArray, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            data: The array to create features from.
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        expand_slices = [slice(None)] * data.ndim
        expand_slices[self.axis] = np.newaxis
        expand_slices = tuple(expand_slices)
        if self.preappend is None:
            slices = [slice(None)] * data.ndim
            slices[self.axis] = 0
            data_ll = np.abs(np.diff(data, axis=self.axis, prepend=data[tuple(slices)][expand_slices]))
        else:
            data_ll = np.abs(np.diff(data, axis=self.axis, prepend=self.preappend))

        slices = [slice(None)] * data.ndim
        slices[self.axis] = -1
        self.preappend = data[tuple(slices)][expand_slices]

        if self.squared_estimator:
            data_ll = data_ll ** 2

        if self.window_len > 0:
            data_ll = convolve(
                data_ll,
                self.window_type(self.window_len).reshape(-1, 1),
                mode='same',
            )

        if self.squared_estimator:
            data_ll = np.sqrt(data_ll)

        # Output
        if isinstance(data, BaseProxyArray):
            data_deep = data.dataless_proxy_leaf_copy()
            data_deep.data = data_ll
            return data_deep
        else:
            return data_ll
