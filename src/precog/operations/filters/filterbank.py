""" filterbank.py

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
from copy import deepcopy
from itertools import chain
from typing import ClassVar, Any

# Third-Party Packages #
from blockobjects import BaseBlock
import numpy as np
from proxyarrays import BaseProxyArray
from scipy.signal import filtfilt

# Local Packages #
from .basefilterbuilder import Filter, BaseFilterBuilder


# Definitions #
# Classes #
class FilterBank(BaseBlock):

    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names: ClassVar[tuple[str, ...]] = ("filter_data",)

    # Attributes #
    axis: int = 0

    sample_rate: float | None = None
    filter_builders: list[BaseFilterBuilder, ...] = []
    filters: list[Filter] = []

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        builders: list | None = None,
        filters: list | None = None,
        sample_rate: float | None = None,
        axis: int | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.filter_builders: list[BaseFilterBuilder, ...] = self.filter_builders.copy()
        self.filters: list[Filter] = self.filters.copy()

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                *args,
                builders=builders,
                filters=filters,
                sample_rate=sample_rate,
                axis=axis,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        builders: list | None = None,
        filters: list | None = None,
        sample_rate: float | None = None,
        axis: int | None = None,
        *args: Any,
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
        if builders is not None:
            self.filter_builders.clear()
            self.filter_builders.extend(builders)

        if filters is not None:
            self.filters.clear()
            self.filters.extend(filters)

        if sample_rate is not None:
            self.sample_rate = sample_rate

        if axis is not None:
            self.axis = axis

        # Construct Parent #
        super().construct(*args, **kwargs)

    # Create Filters
    def create_filters(self):
        self.filters.clear()
        self.filters.extend(chain.from_iterable(
            builder.create_filters_iter(sample_rate=self.sample_rate) for builder in self.filter_builders
        ))

    # Setup
    def setup(
        self,
        sample_rate: float | None = None,
        axis: int | None = None,
        create_filters: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """A method for setting up the object before it runs operation."""
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if axis is not None:
            self.axis = axis

        if create_filters:
            self.create_filters()

    # Evaluate
    def evaluate(self, data: np.ndarray | BaseProxyArray, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            data: The array to remap.
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        # Input
        data_deep = data.dataless_proxy_leaf_copy() if isinstance(data, BaseProxyArray) else None

        # Filtering
        for filter_ in self.filters:
            data = filter_.filter(x=data, axis=self.axis)

        # Output
        if data_deep is None:
            return data
        else:
            data_deep.data = data
            return data_deep
