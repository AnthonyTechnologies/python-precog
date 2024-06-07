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
import timeit

# Third-Party Packages #
from blockobjects.process import DEFAULT_PROCESS_CONTEXT
import pytest
import numpy as np

# Local Packages #
from precog.operations.standardizers import NNMFLineLengthStandardizer


# Definitions #
# Functions #
@pytest.fixture
def tmp_dir(tmpdir):
    """A pytest fixture that turn the tmpdir into a Path object."""
    return pathlib.Path(tmpdir)


# Classes #
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


class TestNNMFLineLengthStandardizer(ClassTest):

    async def start_async(self, data, *args, **kwargs):
        # Create Block Group
        standardizer = NNMFLineLengthStandardizer(**kwargs)
        # Start Block Group
        await standardizer.start_async()

        # Send Data
        await standardizer.inputs.put_item_async("data", data)

        # Get Output
        outputs_1 = await standardizer.outputs.get_all_async()

        # Stop Block
        await standardizer.stop_async()

        assert np.all(outputs_1["features"][1] >= 0)

    def test_random_execute(self):
        DEFAULT_PROCESS_CONTEXT.select_context("multiprocessing")

        # Create Test Data
        samples = 102400
        channels = 512
        t_data = np.random.normal(loc=7, scale=3, size=(samples, channels))

        run(self.start_async(
            data=t_data,
            forget_factor=10**-6,
            mean=np.expand_dims(t_data[0, :], 0),
            threshold=1,
        ))

    def test_random_execute_decay_mean(self):
        DEFAULT_PROCESS_CONTEXT.select_context("multiprocessing")

        # Create Test Data
        samples = 102400
        channels = 512
        t_data = np.random.normal(loc=7, scale=3, size=(samples, channels))

        run(self.start_async(
            data=t_data,
            forget_factor=10**-6,
            shift_scale="shift_decaying_mean",
        ))


# Main #
if __name__ == "__main__":
    pytest.main(["-v", "-s"])
