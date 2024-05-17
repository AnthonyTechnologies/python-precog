""" basefilterbuilder.py

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
from collections.abc import Generator
from typing import Any, NamedTuple

# Third-Party Packages #
from baseobjects import BaseObject

# Local Packages #


# Definitions #
# Classes #
class Filter(NamedTuple):
    filt: Any
    kwargs: dict[str, Any]

    def filter(self, x: Any, **kwargs):
        return self.filt(x=x, **(self.kwargs | kwargs))


class BaseFilterBuilder(BaseObject):
    # New Attributes #
    sample_rate: float | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        sample_rate: float | None = None,
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
                sample_rate=sample_rate,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        sample_rate: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            sample_rate: The sample rate of the data to filter.
            *args: Arguments for inheritance.
            **kwargs: Keyword arguments for inheritance.
        """
        if sample_rate is not None:
            self.sample_rate = sample_rate

        # Construct Parent #
        super().construct(*args, **kwargs)

    # Create Filters
    @abstractmethod
    def create_filters_iter(self, sample_rate: float | None = None, **kwargs: Any) -> Generator[Filter, None, None]:
        """An abstract method for creating a Generator of Filters.

        Args:
            sample_rate: The sample rate of the filters to create.
            **kwargs: The keyword arguments to create filters

        Returns:
            The Filters created.
        """

    def create_filters(self, sample_rate: float | None = None, **kwargs: Any) -> tuple[Filter, ...]:
        return tuple(self.create_filters_iter(sample_rate=sample_rate, **kwargs))
