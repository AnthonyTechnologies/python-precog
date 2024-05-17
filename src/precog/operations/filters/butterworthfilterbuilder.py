""" lowpassfilter.py

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
from typing import ClassVar, Any

# Third-Party Packages #
from scipy.signal import buttord, butter, sosfiltfilt

# Local Packages #
from .basefilterbuilder import Filter, BaseFilterBuilder


# Definitions #
# Classes #
class ButterworthFilterBuilder(BaseFilterBuilder):
    # Attributes #
    butter_types: set[str] = {'bandpass', 'lowpass', 'highpass', 'bandstop'}
    butter_type: str = "bandpass"

    pass_frequency: float = 0.0
    stop_frequency: float = 0.0
    gpass: float = 3
    gstop: float = 60.0
    analog: bool = False

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        pass_frequency: float | None = None,
        stop_frequency: float | None = None,
        gpass: float | None = None,
        gstop: float | None = None,
        analog: bool | None = None,
        butter_type: str | None = None,
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
                pass_frequency=pass_frequency,
                stop_frequency=stop_frequency,
                gpass=gpass,
                gstop=gstop,
                analog=analog,
                butter_type=butter_type,
                sample_rate=sample_rate,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        pass_frequency: float | None = None,
        stop_frequency: float | None = None,
        gpass: float | None = None,
        gstop: float | None = None,
        analog: bool | None = None,
        butter_type: str | None = None,
        sample_rate: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            pass_frequency: The passband frequency.
            stop_frequency: The stopband frequency.
            gpass: The maximum loss in the passband (dB).
            gstop: The minimum attenuation in the stopband (dB).
            analog: Determines if the filter will be analog or digital.
            butter_type: The type of butterworth filter.
            sample_rate: The sample rate of the data to filter.
            *args: Arguments for inheritance.
            **kwargs: Keyword arguments for inheritance.
        """
        if pass_frequency is not None:
            self.pass_frequency = pass_frequency

        if stop_frequency is not None:
            self.stop_frequency = stop_frequency

        if gpass is not None:
            self.gpass = gpass

        if gstop is not None:
            self.gstop = gstop

        if analog is not None:
            self.analog = analog

        if butter_type is not None:
            self.butter_type = butter_type

        # Construct Parent #
        super().construct(*args, sample_rate=sample_rate, **kwargs)

    # Create Filters
    def create_filters_iter(self, sample_rate: float | None = None, **kwargs: Any) -> Generator[Filter, None, None]:
        """Creates a generator that yields Filter objects.

        Args:
            sample_rate: The sample rate of the filter.
            **kwargs: Additional keyword arguments.

        Returns:
            A generator that yields Filter objects.
        """
        if sample_rate is not None:
            self.sample_rate = sample_rate

        if self.stop_frequency > (nq := self.sample_rate / 2):
            pass_frequency = nq * self.pass_frequency / self.stop_frequency
            stop_frequency = nq
        else:
            pass_frequency = self.pass_frequency
            stop_frequency = self.stop_frequency

        # Get butterworth filter parameters
        ford, wn = buttord(
            wp=pass_frequency,   # Passband
            ws=stop_frequency,   # Stopband
            gpass=self.gpass,    # 3dB corner at pass band
            gstop=self.gstop,    # 60dB min. attenuation at stop band
            analog=self.analog,  # Digital filter
            fs=self.sample_rate,
        )

        # Design the filter using second-order sections to ensure better stability
        sos = butter(ford, wn, btype=self.butter_type, output='sos', fs=self.sample_rate)

        return (Filter(sosfiltfilt, {"sos": sos}) for _ in (0,))
