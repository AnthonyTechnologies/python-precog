""" notchfilter.py

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
from scipy.signal import iirnotch, filtfilt

# Local Packages #


# Definitions #
# Classes #
class NotchFilter(BaseBlock):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data", )
    default_output_names: ClassVar[tuple[str, ...]] = ("filter_data",)

    # Attributes #
    axis: int = 0

    sample_rate: float = 1.0
    notch_frequency: float = 60.0
    bandwidth: float = 2.0
    notch_harmonics: bool = True
    _nyquist_frequency: float = 0.0
    _harmonics: np.ndarray | None = None

    filters: list = []

    @property
    def nyquist_frequency(self) -> float:
        self._nyquist_frequency = self.sample_rate / 2
        return self._nyquist_frequency

    @property
    def harmonics(self):
        self._harmonics = np.arange(self.notch_frequency, self.nyquist_frequency, self.notch_frequency)
        return self._harmonics

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        sample_rate: float | None = None,
        notch_frequency: float | None = None,
        bandwidth: float | None = None,
        notch_harmonics: bool | None = True,
        axis: int | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.axis: int = 0

        self.sample_rate: float = 1.0
        self.notch_frequency: float = 60.0
        self.bandwidth: float = 2.0
        self.notch_harmonics: bool = True
        self._nyquist_frequency: float = 0.0
        self._harmonics: np.ndarray | None = None

        self.filters: list = self.filters.copy()

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                *args,
                sample_rate=sample_rate,
                notch_frequency=notch_frequency,
                bandwidth=bandwidth,
                notch_harmonics=notch_harmonics,
                axis=axis,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        sample_rate: float | None = None,
        notch_frequency: float | None = None,
        bandwidth: float | None = None,
        notch_harmonics: bool | None = True,
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
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if notch_frequency is not None:
            self.notch_frequency = notch_frequency

        if bandwidth is not None:
            self.bandwidth = bandwidth

        if notch_harmonics is not None:
            self.notch_harmonics = notch_harmonics

        if axis is not None:
            self.axis = axis

        # Construct Parent #
        super().construct(*args, **kwargs)

    # Create Filters
    def create_filters(self):
        self.filters.clear()
        harmonics = self.harmonics
        freq_list = harmonics[(harmonics - self.bandwidth) > 0]
        freq_list = freq_list[(freq_list + self.bandwidth) < self.nyquist_frequency]
        if self.notch_harmonics:
            for ff in freq_list:
                self.filters.append(iirnotch(ff, ff / self.bandwidth, fs=self.sample_rate))
        else:
            ff = freq_list[0]
            self.filters.append(iirnotch(ff, ff / self.bandwidth, fs=self.sample_rate))

    # Setup
    def setup(
        self,
        sample_rate: float | None = None,
        notch_frequency: float | None = None,
        bandwidth: float | None = None,
        notch_harmonics: bool | None = True,
        axis: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """A method for setting up the object before it runs operation."""
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if notch_frequency is not None:
            self.notch_frequency = notch_frequency

        if bandwidth is not None:
            self.bandwidth = bandwidth

        if notch_harmonics is not None:
            self.notch_harmonics = notch_harmonics

        if axis is not None:
            self.axis = axis

        self.create_filters()

    # Evaluate
    def evaluate(self, data: np.ndarray | None = None, *args, **kwargs: Any) -> Any:
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
            data = filtfilt(filter_[0], filter_[1], data, axis=self.axis)

        # Output
        if data_deep is None:
            return data
        else:
            data_deep.data = data
            return data_deep
