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
from asyncio import run, sleep
import datetime
import pathlib
import timeit

# Third-Party Packages #
import pytest
import numpy as np
import torch
from blockobjects.process import DEFAULT_PROCESS_CONTEXT
from proxyarrays import TimeSeriesProxy, BlankTimeAxis, ContainerTimeAxis, ContainerTimeSeries
from mxbids import Subject

# Local Packages #
from src.precog.operations import ProxyArrayStreamer, CDFSStreamer
from src.precog.models import EnsembleModel
from src.precog.models.torch import NNMFDTorchModel
from src.precog.basis.torch import NonNegativeBasis
from src.precog.pipelines import SpikeDetector


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


class TestSpikeDetector(ClassTest):
    # subjects_root = pathlib.Path("/data_store0/human/converted_clinical")
    # subjects_root = pathlib.Path("//JasperNAS/root_store/subjects")
    subjects_root = pathlib.Path("/Users/changlab/Documents/jaspernas/root_store/updated_subjects")
    subject_id = "EC0283"

    def closest_square(self, n):
        n = int(n)
        i = int(np.ceil(np.sqrt(n)))
        while True:
            if (n % i) == 0:
                break
            i += 1
        assert n == (i * (n // i))
        return i, n // i

    def make_bipolar(self, montage):
        groups = []
        g_name = montage["name"][0].strip("1234567890")
        f_index = 0
        for i, c_name in enumerate(montage["name"]):
            if (new_name := c_name.strip("1234567890")) != g_name:
                groups.append((g_name, montage[f_index:i]))
                g_name = new_name
                f_index = i
        groups.append((g_name, montage[f_index:len(montage)]))

        pb_groups = {}
        remap_contacts = []
        cn_contacts = 0
        for (name, group) in groups:
            type_ = tuple(group["group"])[0]
            c_names = tuple(group["name"])
            n_contact = len(group)
            n_row, n_col = self.closest_square(n_contact) if 'grid' in type_ else (n_contact, 1)

            CA = np.arange(n_contact).reshape((n_row, n_col), order='F')

            pb_groups[name] = bp_contacts = []

            if n_row > 1:
                for bp1, bp2 in zip(CA[:-1, :].flatten(), CA[1:, :].flatten()):
                    bp_contacts.append((c_names[bp1], c_names[bp2], bp1, bp2))
                    remap_contacts.append((bp1 + cn_contacts, bp2 + cn_contacts))

            if n_col > 1:
                for bp1, bp2 in zip(CA[:, :-1].flatten(), CA[:, 1:].flatten()):
                    bp_contacts.append((c_names[bp1], c_names[bp2], bp1, bp2))
                    remap_contacts.append((bp1 + cn_contacts, bp2 + cn_contacts))

            cn_contacts += n_contact

        remap = np.zeros((len(montage), len(remap_contacts)))
        for i, (a, c) in enumerate(remap_contacts):
            remap[a, i] = 1
            remap[c, i] = -1

        return pb_groups, remap

    def test_construction(self):
        detector = SpikeDetector(preprocessing={"sample_rate": 1024})
        bases = detector.model.get_bases()
        state_variables = detector.model.get_state_variables()
        assert detector is not None

    def create_time_series(self, sample_rate, channels):
        time_series = TimeSeriesProxy()
        generator = BlankTimeAxis(start=0, sample_rate=sample_rate, shape=(600000,), precise=True)
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

    def create_proxy_streamer(self, sample_rate, channels, slices):
        data = self.create_time_series(sample_rate, channels)
        return ProxyArrayStreamer(data, setup_kwargs={"slices": slices})

    async def start_async(self, spike_detector, *args, **kwargs):
        # Start Block Group
        await spike_detector.start_async()

        # Get Output
        await sleep(120)

        # Stop Block
        await spike_detector.stop_async()

    def test_proxy_streamer(self):
        DEFAULT_PROCESS_CONTEXT.select_context("multiprocessing")
        # Create Data Streamer
        sample_rate = 1024.0
        channels = 512
        streamer = self.create_proxy_streamer(sample_rate, 512, [slice(0, int(sample_rate))])
        new_map = np.identity(channels)

        # Create Tensor Info
        window_size = int(sample_rate * 0.250)
        n_motifs = 10
        w_size = (channels, n_motifs, window_size)
        h_size = (1, n_motifs, int(sample_rate * 10) - window_size + 1)

        # Create Models
        submodels = {}

        submodels["first_model"] = NNMFDTorchModel(
            architecture={"W": NonNegativeBasis(size=w_size), "H": NonNegativeBasis(size=h_size)},
            trainer={"state_variables": {
                "w_modifier": {
                    "theta": 3600,
                    "beta": 1.0,
                    "penalty": 1.0,
                    "pos": torch.zeros(size=w_size),
                    "neg": torch.zeros(size=w_size),
                },
                "h_modifier": {
                    "theta": 0,
                    "beta": 1.0,
                    "penalty": 1.0,  # Todo: Add this later
                    "pos": torch.zeros(size=h_size),  # torch.zeros_like(theta, memory_format=torch.preserve_format)
                    "neg": torch.zeros(size=h_size),
                },
            }},
        )
        submodels["first_model"].trainer.will_proxy = True

        model = EnsembleModel(submodels=submodels)

        # Create Pipeline
        spike_detector = SpikeDetector(
            model=model,
            streamer=streamer,
            remapper={"map_matrix": new_map},
            preprocessing={"sample_rate": sample_rate},
            standardizer={"forget_factor": 10**-6, "burn_in": 10},  # Todo: Handle burn in (automatic)
            time_buffer={
                "window_time": 10,
                "window_samples": int(sample_rate * 10),
                "window_tolerance": 0.0005,
                "step_time": 0.250,
                "step_samples": int(sample_rate * 0.250),
                "step_tolerance": 0.0005,
                "sample_rate": sample_rate,
            },
        )

        # Evaluate
        run(self.start_async(spike_detector))

        assert True

    def test_evaluate_stream(self):
        DEFAULT_PROCESS_CONTEXT.select_context("multiprocessing")

        # Select Subject
        bids_subject = Subject(name=self.subject_id, parent_path=self.subjects_root)
        session = bids_subject.sessions["clinicalintracranial"]
        ieeg = session.modalities["ieeg"]
        cdfs = ieeg.components["cdfs"].get_cdfs()
        proxy = cdfs.components["contents"].require_contents_proxy()

        # Remap Channels
        sample_rate = proxy.sample_rates[1]
        montage = ieeg.load_electrodes()
        b_groups, remap = self.make_bipolar(montage)
        new_map = np.zeros((276 if remap.shape[0] > 128 else 148, remap.shape[1]), dtype="f4")
        new_map[:remap.shape[0], :remap.shape[1]] = remap

        # max_channels = np.array(cdfs.data.shapes).max(0)[1]
        # used_channels = len(montage["name"][:-4])
        # remap = np.zeros((max_channels, used_channels))
        # remap[:used_channels, :] = np.identity(used_channels)  # Remap Channels from Montage

        # Create Tensor Info
        window_size = int(sample_rate * 0.250)
        n_motifs = 10
        w_size = (remap.shape[1], n_motifs, window_size)
        h_size = (1, n_motifs, int(sample_rate * 10) - window_size + 1)

        # Create Models
        submodels = {}

        submodels["first_model"] = NNMFDTorchModel(
            architecture={"W": NonNegativeBasis(size=w_size), "H": NonNegativeBasis(size=h_size)},
            trainer={"state_variables": {
                "w_modifier": {
                    "theta": 3600,
                    "beta": 1.0,
                    "penalty": 1.0,
                    "pos": torch.zeros(size=w_size),
                    "neg": torch.zeros(size=w_size),
                },
                "h_modifier": {
                    "theta": 0,
                    "beta": 1.0,
                    "penalty": 1.0,  # Todo: Add this later
                    "pos": torch.zeros(size=h_size),  # torch.zeros_like(theta, memory_format=torch.preserve_format)
                    "neg": torch.zeros(size=h_size),
                },
            }},
        )
        submodels["first_model"].trainer.will_proxy = True

        model = EnsembleModel(submodels=submodels)

        # Create Pipeline
        spike_detector = SpikeDetector(
            model=model,
            streamer=CDFSStreamer(cdfs=cdfs),
            remapper={"map_matrix": new_map},
            preprocessing={"sample_rate": sample_rate},
            standardizer={"threshold": 3, "forget_factor": 10 ** -6, "burn_in": 10},
            time_buffer={
                "window_time": 10,
                "window_samples": int(sample_rate * 10),
                "window_tolerance": 0.0005,
                "step_time": 0.250,
                "step_samples": int(sample_rate * 0.250),
                "step_tolerance": 0.0005,
                "sample_rate": sample_rate,
            },
        )

        # Select Time Range
        start = datetime.datetime(1970, 1, 7, 0, 5, 0, tzinfo=datetime.timezone.utc)
        stop = datetime.datetime(1970, 1, 7, 1, 5, 10, tzinfo=datetime.timezone.utc)

        streamer = spike_detector.blocks["streamer"]
        streamer.setup_kwargs.update(start=start, stop=stop, step=10, approx=True, tails=True)

        # Evaluate
        run(self.start_async(spike_detector))

        assert True


# Main #
if __name__ == "__main__":
    # pytest.main(["-v", "-s"])
    t = TestSpikeDetector()
    t.test_evaluate_stream()
