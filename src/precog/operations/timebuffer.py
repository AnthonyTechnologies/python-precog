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
from collections.abc import Iterable
from collections import deque
from datetime import datetime, date, timedelta
from typing import ClassVar, Any
from uuid import uuid4

# Third-Party Packages #
from blockobjects import BaseBlock
from blockobjects.io import IdentifiedItem
from dspobjects.time import Timestamp, nanostamp
import numpy as np
from proxyarrays import BaseProxyArray, BaseTimeAxis, BaseTimeSeries, TimeSeriesProxy, ContainerTimeSeries

# Local Packages #


# Definitions #
# Classes #
class TimeBuffer(BaseBlock):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data", "time_axis")
    default_required_input: ClassVar[tuple[str, ...] | None] = ()
    default_optional_input: ClassVar[dict[str, Any]] = {"data": None, "time_axis": None}
    default_output_names: ClassVar[tuple[str, ...]] = ("buffer_data",)

    # Attributes #
    # IO
    put_output_method = "_put_multiple_output"
    put_output_method_async = "_put_multiple_output_async"

    # Buffer
    axis: int = 0
    data_buffer: deque
    time_buffer: deque
    buffer: BaseProxyArray

    current_start_ns: np.uint64 | None = None

    window_samples: int
    window_time_ns: np.uint64
    window_tolerance_ns: np.uint64

    step_samples: int
    step_time_ns: np.uint64
    step_tolerance_ns: np.uint64

    sample_period_ns: np.uint64

    # Properties #
    @property
    def current_start(self) -> Timestamp | None:
        return None if self.current_start_ns is None else Timestamp.fromnanostamp(self.current_start_ns)

    @current_start.setter
    def current_start(self, value: datetime | date | float | int | np.dtype) -> None:
        self.current_start_ns = nanostamp(value)

    @property
    def window_time(self) -> float:
        return self.window_time_ns / 1e9

    @window_time.setter
    def window_time(self, value: timedelta | float | int | np.dtype) -> None:
        self.window_time_ns = nanostamp(value)

    @property
    def window_tolerance(self) -> float:
        return self.window_tolerance_ns / 1e9

    @window_tolerance.setter
    def window_tolerance(self, value: timedelta | float | int | np.dtype) -> None:
        self.window_tolerance_ns = nanostamp(value)

    @property
    def step_time(self) -> float:
        return self.step_time_ns / 1e9

    @step_time.setter
    def step_time(self, value: timedelta | float | int | np.dtype) -> None:
        self.step_time_ns = nanostamp(value)

    @property
    def step_tolerance(self) -> float:
        return self.step_tolerance_ns / 1e9

    @step_tolerance.setter
    def step_tolerance(self, value: timedelta | float | int | np.dtype) -> None:
        self.step_tolerance_ns = nanostamp(value)

    @property
    def sample_period(self) -> float:
        return self.sample_period_ns / 1e9

    @sample_period.setter
    def sample_period(self, value: timedelta | float | int | np.dtype) -> None:
        self.sample_period_ns = nanostamp(value)

    @property
    def sample_rate(self) -> float:
        return 1e9 / self.sample_period_ns

    @sample_rate.setter
    def sample_rate(self, value: float | int | np.dtype) -> None:
        self.sample_period_ns = np.uint64(1e9 / value)

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        start: datetime | date | float | int | np.dtype | None = None,
        window_time: timedelta | float | int | np.dtype | None = None,
        window_samples: int | None = None,
        window_tolerance: timedelta | float | int | np.dtype | None = None,
        step_time: timedelta | float | int | np.dtype | None = None,
        step_samples: int | None = None,
        step_tolerance: timedelta | float | int | np.dtype | None = None,
        sample_rate: float | int | np.dtype | None = None,
        sample_period: timedelta | float | int | np.dtype | None = None,
        axis: int | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Attributes #
        self.time_buffer = deque()
        self.data_buffer = deque()
        self.buffer = TimeSeriesProxy()

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                start=start,
                window_time=window_time,
                window_samples=window_samples,
                window_tolerance=window_tolerance,
                step_time=step_time,
                step_samples=step_samples,
                step_tolerance=step_tolerance,
                sample_rate=sample_rate,
                sample_period=sample_period,
                axis=axis,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        start: datetime | date | float | int | np.dtype | None = None,
        window_time: timedelta | float | int | np.dtype | None = None,
        window_samples: int | None = None,
        window_tolerance: timedelta | float | int | np.dtype | None = None,
        step_time: timedelta | float | int | np.dtype | None = None,
        step_samples: int | None = None,
        step_tolerance: timedelta | float | int | np.dtype | None = None,
        sample_rate: float | int | np.dtype | None = None,
        sample_period: timedelta | float | int | np.dtype | None = None,
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
        if start is not None:
            self.current_start = start

        if window_time is not None:
            self.window_time = window_time

        if window_samples is not None:
            self.window_samples = window_samples

        if window_tolerance is not None:
            self.window_tolerance = window_tolerance

        if step_time is not None:
            self.step_time = step_time

        if step_samples is not None:
            self.step_samples = step_samples

        if step_tolerance is not None:
            self.step_tolerance = step_tolerance

        if sample_rate is not None:
            self.sample_rate = sample_rate

        if sample_period is not None:
            self.sample_period = sample_period

        if axis is not None:
            self.axis = axis
            self.buffer.axis = axis

        # Construct Parent #
        super().construct(*args, **kwargs)

    # IO
    def format_output(
        self,
        outputs: Any,
        ids: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> list[dict[str, Any]] | None:
        keys = self.outputs.order
        if len(keys) == 1:
            return [{keys[0]: IdentifiedItem((uuid4().bytes,), out)} for out in outputs]
        else:
            new_outputs = []
            for i in range(len(outputs[0])):
                ids = (uuid4().bytes,)
                new_outputs.append({k: IdentifiedItem(ids, v) for k, v in zip(keys, [out[i] for out in outputs])})
            return new_outputs

    # Setup
    def setup(self, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        pass

    # Evaluate
    def evaluate(
        self,
        data: np.ndarray | BaseTimeSeries | None = None,
        time_axis: np.ndarray | BaseTimeAxis | None = None,
        *args,
        input_ids: dict[str, tuple[bytes, ...]] | None = None,
        **kwargs: Any,
    ) -> Any:
        # Asynchronous Buffer
        items = deque()
        if isinstance(data, BaseTimeSeries):
            items.append(data)
            d = None
        else:
            d = data

        if d is not None or time_axis is not None:
            matched_items = self.parse_async_buffer(data=d, time_axis=time_axis, input_ids=input_ids)
            if matched_items is not None:
                items.extend(matched_items)

        # Synchronization Buffer
        self.insert_into_buffer(items)

        return self.dispense_data()

    # Asynchronous Buffer
    def check_ids(self, a: set, b: set) -> bool:
        return not a.isdisjoint(b)

    def check_buffer_ids(
        self,
        compare_buffer: deque,
        store_buffer: deque,
        ids: set,
        item: Any,
    ) -> np.ndarray | BaseProxyArray | None:
        # Pops and returns the first match in the buffer
        for i, d in enumerate(compare_buffer):
            if self.check_ids(ids, d[0]):
                del compare_buffer[i]
                return d[1]

        # Add the item and its IDs to buffer if there is no match
        store_buffer.append((ids, item))

    def parse_async_buffer(
        self,
        data: np.ndarray | None = None,
        time_axis: np.ndarray | BaseTimeAxis | None = None,
        input_ids: dict[str, tuple[bytes, ...]] | None = None,
    ) -> tuple[BaseTimeSeries] | None:
        # Get IDs
        data_ids = set(input_ids.get("data", ()))
        time_ids = set(input_ids.get("time_axis", ()))

        # Check if both data and time_axis match and return them as a time series if they do
        if data_ids and time_ids and self.check_ids(data_ids, time_ids):
            return (ContainerTimeSeries(data=data, time_axis=time_axis),)

        d = deque()
        # Check if data matches a time_axis in the buffer, otherwise add it to the buffer
        if data is not None:
            matched_time = self.check_buffer_ids(self.time_buffer, self.data_buffer, data_ids, data)
            if matched_time is not None:
                d.append(ContainerTimeSeries(data=data, time_axis=matched_time))

        # Check if time_axis matches a data in the buffer, otherwise add it to the buffer
        if time_axis is not None:
            matched_data = self.check_buffer_ids(self.data_buffer, self.time_buffer, time_ids, time_axis)
            if matched_data is not None:
                d.append(ContainerTimeSeries(data=matched_data, time_axis=time_axis))

        # Return any created time series
        if d:
            return tuple(d)

    # Synchronization Buffer
    def insert_into_buffer(self, items: Iterable[BaseTimeSeries, ...]) -> None:
        for item in items:
            if self.buffer.end_nanostamp is None or item.start_nanostamp > self.buffer.end_nanostamp:
                self.buffer.append(item)
            else:
                index = int(np.searchsorted(self.buffer.get_start_nanostamps(), item.start_nanostamp))
                self.buffer.proxies.insert(index, item)

    def dispense_data(self) -> BaseTimeSeries | None:
        if self.current_start_ns is None:
            self.current_start_ns = self.buffer.start_nanostamp

        n_samples = self.buffer.get_length()
        start_index = self.buffer.find_time_index(self.current_start_ns, tails=True)[0]

        data = deque()
        while n_samples - start_index >= self.window_samples:
            # Validate Window
            start_nanostamp = self.buffer.get_nanostamp(start_index)
            end_index = start_index + self.window_samples
            nanostamps = self.buffer.nanostamp_slice(start_index, end_index)
            t_diff = abs(nanostamps[-1] + self.sample_period_ns - nanostamps[0] - self.window_time_ns)
            if t_diff <= self.window_tolerance_ns:
                slices = [slice(None)] * self.buffer.ndim
                slices[self.axis] = slice(start_index, start_index + self.window_samples)
                data.append(self.buffer.return_proxy_leaf(
                    data=self.buffer.slices_array(slices),
                    time_axis=self.buffer.nanostamp_slice(start_index, start_index + self.window_samples),
                ))
            elif self.allow_window_drift:
                start_index += 1
                continue

            # Check if there is enough data to step
            next_index = start_index + self.step_samples
            if n_samples < next_index + self.window_samples:
                start_nanostamp = self.buffer.get_nanostamp(start_index) + self.step_time_ns
                break

            # Validate Step
            step_nanostamp = self.buffer.get_nanostamp(next_index) - start_nanostamp
            if step_nanostamp < self.step_tolerance_ns or abs(step_nanostamp % self.step_time_ns) > self.step_tolerance_ns:
                next_index = self.buffer.find_time_index(start_nanostamp + self.step_time, True)[0]
                if next_index == start_index:
                    next_index += 1
            start_index = next_index

        self.current_start_ns = start_nanostamp
        for i in range(len(self.buffer.proxies)):
            if self.buffer.proxies[0].end_nanostamp < start_nanostamp:
                self.buffer.proxies.pop(0)
            else:
                break

        return data

