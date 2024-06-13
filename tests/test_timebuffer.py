#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" test_hdf5objects.py
Description:
"""
# Package Header #
from src.precog.header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from asyncio import run
import datetime
import pathlib
from typing import ClassVar, Any
import timeit

# Third-Party Packages #
from blockobjects import BlockGroup
from blockobjects.process import DEFAULT_PROCESS_CONTEXT
import pytest
import numpy as np
from proxyarrays import TimeSeriesProxy, BlankTimeAxis, ContainerTimeAxis, ContainerTimeSeries

# Local Packages #
from src.precog.operations import ProxyArrayStreamer, TimeBuffer


# Definitions #
# Functions #
@pytest.fixture
def tmp_dir(tmpdir):
    """A pytest fixture that turn the tmpdir into a Path object."""
    return pathlib.Path(tmpdir)


# Classes #
class BlockGroupTest(BlockGroup):
    default_output_names: ClassVar[tuple[str, ...]] = ("group_result",)

    # Instance Methods #
    # Blocks
    def create_blocks(
        self,
        data: np.ndarray | None = None,
        window_time: float | int | np.dtype | None = None,
        window_samples: int | None = None,
        window_tolerance: float | int | np.dtype | None = None,
        step_time: float | int | np.dtype | None = None,
        step_samples: int | None = None,
        step_tolerance: float | int | np.dtype | None = None,
        sample_rate: float | int | np.dtype | None = None,
        slices: slice | None = None,
        *args: Any,
        override: bool = False,
        **kwargs: Any,
    ) -> None:
        """Creates the inner blocks.

        Args:
            *args: The arguments for creating the inner blocks.
            override: Determines if the inner blocks will be overridden.
            **kwargs: The keyword arguments for creating the inner blocks.
        """
        # Create Blocks
        self.blocks["generator"] = ProxyArrayStreamer(data, setup_kwargs={"slices": slices})
        self.blocks["time_buffer"] = TimeBuffer(
            window_time=window_time,
            window_samples=window_samples,
            window_tolerance=window_tolerance,
            step_time=step_time,
            step_samples=step_samples,
            step_tolerance=step_tolerance,
            sample_rate=sample_rate,
        )

    # IO
    def link_inner_io(self, *args: Any, **kwargs: Any) -> None:
        """Links the inner blocks' IO.

        Args:
            *args: The arguments for creating linking the inner blocks' IO.
            **kwargs: The keyword arguments for creating linking the inner blocks' IO.
        """
        # Get Blocks
        generator = self.blocks["generator"]
        time_buffer = self.blocks["time_buffer"]

        # Group Inputs
        # No inputs to group because it generates its own data.

        # Inner Block IO
        generator.outputs.link_forward("data", time_buffer.inputs, "data")

        # Group Outputs
        time_buffer.outputs.link_forward("buffer_data", self.outputs, "group_result")


class ClassTest:
    """Default class tests that all classes should pass."""

    class_ = None
    timeit_runs = 2
    speed_tolerance = 200

    def get_log_lines(self, tmp_dir, logger_name):
        path = tmp_dir.joinpath(f"{logger_name}.log")
        with path.open() as f_object:
            lines = f_object.readlines()
        return lines


class TestCDFSStreamer(ClassTest):

    def create_time_series(self, sample_rate, channels):
        time_series = TimeSeriesProxy()
        generator = BlankTimeAxis(start=0, sample_rate=sample_rate, shape=(100000,), precise=True)
        segments = (
            (0, int(sample_rate * 100)),
            (0, int(sample_rate * 10)),
            (0, int(sample_rate * 0.5)),
            (0, int(sample_rate * 0.5)),
            (0, int(sample_rate * 100)),
            (100, int(sample_rate * 100)),
            (0, int(sample_rate * 100)),
            (0, int(sample_rate * 100.5)),
            (0, int(sample_rate * 100)),
        )
        end = 0

        for gap, length in segments:
            start = end + gap
            end = start + length
            times = generator[start:end]
            time_axis = ContainerTimeAxis(data=times, sample_rate=sample_rate, precise=True)
            data = np.random.rand(len(times), channels) - 0.5
            time_series.proxies.append(ContainerTimeSeries(data=data, time_axis=time_axis))

        return time_series

    async def start_async(self, *args, **kwargs):
        # Create Block Group
        time_group = BlockGroupTest(**kwargs)
        # Start Block Group
        await time_group.start_async()

        # Get Output
        outputs_1 = await time_group.outputs.get_all_async()
        outputs_2 = await time_group.outputs.get_all_async()
        outputs_3 = await time_group.outputs.get_all_async()
        outputs_4 = await time_group.outputs.get_all_async()

        # Stop Block
        await time_group.stop_async()

        assert outputs_1


    def test_evaluate_stream(self):
        DEFAULT_PROCESS_CONTEXT.select_context("multiprocessing")

        run(self.start_async(
            create_kwargs={"data": self.create_time_series(1000, 512),
                           "window_time": 1,
                           "window_samples": 1000,
                           "window_tolerance": 0.001,
                           "step_time": 0.5,
                           "step_samples": 500,
                           "step_tolerance": 0.0005,
                           "sample_rate": 1000,
                           "slices": [slice(0, 1000)],
                           },
        ))


# Main #
if __name__ == "__main__":
    # pytest.main(["-v", "-s"])
    t = TestCDFSStreamer()
    t.test_evaluate_stream()
