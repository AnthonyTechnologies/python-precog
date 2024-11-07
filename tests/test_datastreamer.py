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
import datetime
import pathlib
import timeit

# Third-Party Packages #
import pytest
import numpy as np
from mxbids import Subject
from h5py import Dataset
import matplotlib.pyplot as plt

# Local Packages #
from src.precog.operations import CDFSStreamer


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


class TestCDFSStreamer(ClassTest):
    subjects_root = pathlib.Path("/data_store0/human/converted_clinical")
    subject_id = "EC0300"

    def test_evaluate_stream(self):
        from xltektools.xltekmxbids import IEEGXLTEK
        start = datetime.datetime(1970, 1, 7, 0, 5, 0, tzinfo=datetime.timezone.utc)
        stop = datetime.datetime(1970, 1, 7, 0, 9, 10, tzinfo=datetime.timezone.utc)

        bids_subject = Subject(name=self.subject_id, parent_path=self.subjects_root)
        session = bids_subject.sessions["clinicalintracranial"]
        cdfs = session.modalities["ieeg"].require_cdfs()

        streamer = CDFSStreamer(cdfs=cdfs)
        streamer.setup(start=start, stop=stop, step=10, approx=True, tails=True)

        outs = []
        while (segment := streamer.evaluate()) is not None:
            outs.append(segment)

        shapes = [out.shape for out in outs]

        iter_concat = np.concatenate(outs, axis=0)
        t_iter = np.arange(0, iter_concat.shape[0])

        fig, ax = plt.subplots()
        ax.set(xlabel='time (s)', ylabel='voltage (uV)', title='Iter Data')
        ax.plot(t_iter, iter_concat[:, 0])

        plt.show()

        assert True

    def test_evaluate_stream_consistency(self):
        from xltektools.xltekmxbids import IEEGXLTEK
        start = datetime.datetime(1970, 1, 7, 0, 1, 0, tzinfo=datetime.timezone.utc)
        stop = datetime.datetime(1970, 1, 7, 0, 1, 30, tzinfo=datetime.timezone.utc)

        bids_subject = Subject(name=self.subject_id, parent_path=self.subjects_root)
        session = bids_subject.sessions["clinicalintracranial"]
        cdfs = session.modalities["ieeg"].require_cdfs()

        streamer = CDFSStreamer(cdfs=cdfs)
        streamer.setup(start=start, stop=stop, step=1, approx=True, tails=True)

        all_data = cdfs.data.find_data_slice(start=start, stop=stop)
        data = np.array(all_data[0])

        outs = []
        while (seg := streamer.evaluate()) is not None:
            outs.append(seg)

        iter_concat = np.concatenate(outs, axis=0)

        t_iter = np.arange(0, iter_concat.shape[0])
        t_data = np.arange(0, data.shape[0])

        fig, ax = plt.subplots()
        ax.set(xlabel='time (s)', ylabel='voltage (uV)', title='Iter Data Blue, Data Orange')
        ax.plot(t_iter, iter_concat[:, 0])
        ax.plot(t_data, data[:, 0])

        fig, ax = plt.subplots()
        ax.set(xlabel='time (s)', ylabel='voltage (uV)', title='Iter Data')
        ax.plot(t_iter, iter_concat[:, 0])

        fig, ax = plt.subplots()
        ax.set(xlabel='time (s)', ylabel='voltage (uV)', title='Data')
        ax.plot(t_data, data[:, 0])

        plt.show()

        assert True

    def test_evaluate_stream_overlap(self):
        from xltektools.xltekmxbids import IEEGXLTEK
        start_1 = datetime.datetime(1970, 1, 7, 0, 1, 0, tzinfo=datetime.timezone.utc)
        stop_1 = datetime.datetime(1970, 1, 7, 0, 1, 30, tzinfo=datetime.timezone.utc)
        start_2 = datetime.datetime(1970, 1, 7, 0, 1, 10, tzinfo=datetime.timezone.utc)
        stop_2 = datetime.datetime(1970, 1, 7, 0, 1, 40, tzinfo=datetime.timezone.utc)

        bids_subject = Subject(name=self.subject_id, parent_path=self.subjects_root)
        session = bids_subject.sessions["clinicalintracranial"]
        cdfs = session.modalities["ieeg"].require_cdfs()

        streamer = CDFSStreamer(cdfs=cdfs)
        streamer.setup(start=start_1, stop=stop_1, step=1, approx=True, tails=True)

        outs = []
        while (seg := streamer.evaluate()) is not None:
            outs.append(seg)

        iter_concat_1 = np.concatenate(outs, axis=0)

        streamer2 = CDFSStreamer(cdfs=cdfs)
        streamer2.setup(start=start_2, stop=stop_2, step=1, approx=True, tails=True)

        outs2 = []
        while (seg := streamer2.evaluate()) is not None:
            outs2.append(seg)

        iter_concat_2 = np.concatenate(outs2, axis=0)

        t_iter_1 = np.arange(0, 1024*30)
        t_iter_2 = np.arange(10240, 1024*40)

        fig, ax = plt.subplots(2, 1)
        ax[0].set(xlabel='time (s)', ylabel='voltage (uV)', title='Two iterators')
        ax[0].plot(t_iter_1, iter_concat_1[:, 0])
        ax[0].plot(t_iter_2, iter_concat_2[:, 0])

        ax[1].plot(t_iter_2, iter_concat_2[:, 0])
        ax[1].plot(t_iter_1, iter_concat_1[:, 0])

        plt.show()

        assert True


# Main #
if __name__ == "__main__":
    # pytest.main(["-v", "-s"])
    t = TestCDFSStreamer()
    t.test_evaluate_stream()
