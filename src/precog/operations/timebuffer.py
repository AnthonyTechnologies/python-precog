""" timebuffer.py

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
from typing import Any

# Third-Party Packages #
import numpy as np
from proxyarrays import BaseProxyArray, BaseTimeAxis, BaseTimeSeries, TimeSeriesProxy

# Local Packages #
from .operation import BaseOperation


# Definitions #
# Classes #
class TimeBuffer(BaseOperation):
    default_input_names: ClassVar[tuple[str, ...]] = ("data", )
    default_output_names: ClassVar[tuple[str, ...]] = ("buffer_data",)

    # New Attributes #
    axis: int = 1
    buffer: BaseProxyArray | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        axis: int | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Attributes #
        self.buffer = TimeSeriesProxy()

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                axis=axis,
                init_io=init_io,
                sets_up=sets_up,
                setup_kwargs=setup_kwargs,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        axis: int | None = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            axis: The axis to remap along.
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            sets_up: Determines if setup will run during this construction.
            setup_kwargs: The keyword arguments for the setup method.
            **kwargs: Keyword arguments for inheritance.
        """

        if axis is not None:
            self.axis = axis
            self.buffer.axis = axis

        # Construct Parent #
        super().construct(*args, init_io=init_io, sets_up=sets_up, setup_kwargs=setup_kwargs, **kwargs)

    # Setup
    def setup(self, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        pass

    # Evaluate
    def evaluate(self, data: BaseTimeSeries | None = None, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            data: The array to buffer.
            map_matrix: The remap matrix to apply.
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        if data is None:
            return None
        else:
            # Todo: Change This later
            return data if data.shape[0] == 10240 else None

