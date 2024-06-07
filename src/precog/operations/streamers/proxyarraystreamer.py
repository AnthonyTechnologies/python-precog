""" baseoperation.py

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
from collections.abc import Iterable, Generator
from datetime import datetime, timedelta
from decimal import Decimal
from typing import ClassVar, Any

# Third-Party Packages #
from baseobjects.functions import MethodMultiplexer
from blockobjects import BaseBlock
import numpy as np
from proxyarrays import BaseProxyArray, BaseTimeAxis, BaseTimeSeries

# Local Packages #


# Definitions #
# Classes #
class ProxyArrayStreamer(BaseBlock):

    # Class Attributes #
    default_output_names: ClassVar[tuple[str, ...]] = ("data",)
    default_create_generator: str = "create_islices"

    init_setup: ClassVar[bool] = False

    # Attributes #
    will_produce = True

    proxy_array: BaseProxyArray | None = None
    empty_signal: Any = None

    generator: Generator | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        proxy_array: BaseProxyArray | None = None,
        empty_signal: Any = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.create_generator: MethodMultiplexer = MethodMultiplexer(
            instance=self,
            select=self.default_create_generator,
        )

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                proxy_array=proxy_array,
                empty_signal=empty_signal,
                *args,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        proxy_array: BaseProxyArray | None = None,
        empty_signal: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            sets_up: Determines if setup will run during this construction.
            setup_kwargs: The keyword arguments for the setup method.
            **kwargs: Keyword arguments for inheritance.
        """
        if proxy_array is not None:
            self.proxy_array = proxy_array

        if empty_signal is not None:
            self.empty_signal = empty_signal

        # Construct Parent #
        super().construct(*args, **kwargs)

    # Create Generator
    def create_islices(
        self,
        slices: Iterable[slice | int | None] | None = None,
        islice: slice | None = None,
        axis: int | None = None,
        dtype: Any = None,
        proxy: bool | None = None,
    ) -> Generator[BaseProxyArray | np.ndarray, None, None]:
        self.generator = self.proxy_array.islices(slices=slices, islice=islice, axis=axis, dtype=dtype, proxy=proxy)
        return self.generator

    def create_islice_time(
        self,
        start: datetime | float | int | np.dtype | None = None,
        stop: datetime | float | int | np.dtype | None = None,
        step: int | float | timedelta | Decimal | None = None,
        istep: int = 1,
        approx: bool = True,
        tails: bool = False,
    ) -> Generator[BaseTimeSeries, None, None]:
        self.generator = self.proxy_array.find_data_islice_time(
            start=start,
            stop=stop,
            step=step,
            istep=istep,
            approx=approx,
            tails=tails,
        )
        return self.generator

    def create_islice_timeaxis(
        self,
        start: datetime | float | int | np.dtype | None = None,
        stop: datetime | float | int | np.dtype | None = None,
        step: int | float | timedelta | None = None,
        istep: int = 1,
        approx: bool = True,
        tails: bool = True,
    ) -> Generator[BaseTimeAxis, None, None]:
        """Creates a generator which yields nanostamps slices based on times.

        Args:
            start: The start time to begin slicing.
            stop: The last time to end slicing.
            step: The time within each slice.
            istep: The step of each slice.
            approx: Determines if an approximate indices will be given if the time is not present.
            tails: Determines if the first or last times will be give the requested item is outside the axis.

        Returns:
            The generator which yields time axis slices.
        """
        self.generator = self.proxy_array.nanostamp_islice_time(
            start=start,
            stop=stop,
            step=step,
            istep=istep,
            approx=approx,
            tails=tails,
        )
        return self.generator

    # Setup
    def setup(self, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        self.create_generator(*args, **kwargs)

    # Evaluate
    def evaluate(self, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        try:
            return next(self.generator)
        except StopIteration:
            return self.empty_signal
