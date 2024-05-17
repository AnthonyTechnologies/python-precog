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
from collections.abc import Generator
from typing import Any

# Third-Party Packages #
from baseobjects import BaseObject
import numpy as np
from scipy.signal import iirnotch, filtfilt

# Local Packages #
from .basefilterbuilder import Filter, BaseFilterBuilder


# Definitions #
# Classes #
class NotchFilterBuilder(BaseFilterBuilder):
    # New Attributes #
    axis: int = 0

    sample_rate: float = 1.0
    notch_frequency: float = 60.0
    bandwidth: float = 2.0
    notch_harmonics: bool = True
    _nyquist_frequency: float = 0.0
    _harmonics: np.ndarray | None = None

    # Properties #
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
        notch_frequency: float | None = None,
        bandwidth: float | None = None,
        notch_harmonics: bool | None = True,
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
                notch_frequency=notch_frequency,
                bandwidth=bandwidth,
                notch_harmonics=notch_harmonics,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        notch_frequency: float | None = None,
        bandwidth: float | None = None,
        notch_harmonics: bool | None = True,
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

        if notch_frequency is not None:
            self.notch_frequency = notch_frequency

        if bandwidth is not None:
            self.bandwidth = bandwidth

        if notch_harmonics is not None:
            self.notch_harmonics = notch_harmonics

        # Construct Parent #
        super().construct(*args, sample_rate=sample_rate, **kwargs)

    # Create Filters
    def create_filters_iter(self, sample_rate: float | None = None, **kwargs) -> Generator[Filter, None, None]:
        if sample_rate is not None:
            self.sample_rate = sample_rate
        harmonics = self.harmonics
        freq_list = harmonics[(harmonics - self.bandwidth) > 0]
        freq_list = freq_list[(freq_list + self.bandwidth) < self.nyquist_frequency]
        if not self.notch_harmonics:
            freq_list = [freq_list[0]]

        return (Filter(filtfilt, dict(zip(("b", "a"), iirnotch(ff, ff / self.bandwidth, fs=self.sample_rate)))) for ff
                in freq_list)
