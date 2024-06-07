""" normnormalization.py

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
from proxyarrays import BaseProxyArray

# Local Packages #


# Definitions #
# Classes #
class NormNormalization(BaseBlock):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names: ClassVar[tuple[str, ...]] = ("n_data",)

    # New Attributes #
    axis: int | tuple[int, int] | None = 0
    order: int | tuple[int, int] | None = None
    keep_dims: bool = False

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        order: int | tuple[int, int] | None = None,
        keep_dims: bool | None = None,
        axis: int | tuple[int, int] | None = None,
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
                order=order,
                keep_dims=keep_dims,
                axis=axis,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        order: int | tuple[int, int] | None = None,
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

        if order is not None:
            self.order = order

        if keep_dims is not None:
            self.keep_dims = keep_dims

        # Construct Parent #
        super().construct(*args, **kwargs)

    # Setup
    def setup(self, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        pass

    # Evaluate
    def evaluate(self, data: np.ndarray | BaseProxyArray, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        n_data = data / np.linalg.norm(data, ord=self.order, axis=self.axis, keepdims=self.keep_dims)

        # Output
        if isinstance(data, BaseProxyArray):
            data_deep = data.dataless_proxy_leaf_copy()
            data_deep.data = n_data
            return data_deep
        else:
            return n_data
