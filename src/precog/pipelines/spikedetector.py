""" spikedetector.py.py

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
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Callable

# Third-Party Packages #
from blockobjects import BaseBlock, BlockGroup
import numpy as np

# Local Packages #
from ..operations.streamers import CDFSStreamer
from ..operations.remapper import Remapper
from ..operations.preprocessingfilterbank import PreprocessingFilterBank
from ..operations.standardizers import NNMFLineLengthStandardizer
from ..operations.timebuffer import TimeBuffer
from ..models import BaseModel
from ..models.torch import NNMFDTorchModel


# Definitions #
# Classes #
class SpikeDetector(BlockGroup):
    # Attributes #
    streamer_type = CDFSStreamer
    remapper_type = Remapper
    preprocessing_type = PreprocessingFilterBank
    standardizer_type = NNMFLineLengthStandardizer
    time_buffer_type = TimeBuffer
    model_type = NNMFDTorchModel
    detector_type = None

    model: BaseModel | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        model: BaseModel | None = None,
        streamer: BaseBlock | dict[str, Any] | None = None,
        remapper: BaseBlock | dict[str, Any] | None = None,
        preprocessing: BaseBlock | dict[str, Any] | None = None,
        standardizer: BaseBlock | dict[str, Any] | None = None,
        time_buffer: BaseBlock | dict[str, Any] | None = None,
        detector: BaseBlock | dict[str, Any] | None = None,
        *args: Any,
        create_kwargs: dict[str, Any] | None = None,
        link_kwargs: dict[str, Any] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                model=model,
                streamer=streamer,
                remapper=remapper,
                preprocessing=preprocessing,
                standardizer=standardizer,
                time_buffer=time_buffer,
                detector=detector,
                create_kwargs=create_kwargs,
                link_kwargs=link_kwargs,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        model: BaseModel | None = None,
        streamer: BaseBlock | dict[str, Any] | None = None,
        remapper: BaseBlock | dict[str, Any] | None = None,
        preprocessing: BaseBlock | dict[str, Any] | None = None,
        standardizer: BaseBlock | dict[str, Any] | None = None,
        time_buffer: BaseBlock | dict[str, Any] | None = None,
        detector: BaseBlock | dict[str, Any] | None = None,
        *args: Any,
        create_kwargs: dict[str, Any] | None = None,
        link_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            blocks: The dictionary of Block to add to the BlockGroup.
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            sets_up: Determines if setup will run during this construction.
            setup_kwargs: The keyword arguments for the setup method.
            **kwargs: Keyword arguments for inheritance.
        """
        # New Assignment #
        if model is not None:
            self.model = model
        
        # Kwargs for Block Creation
        create_kwargs = {} if create_kwargs is None else create_kwargs
        
        if isinstance(streamer, BaseBlock):
            self.blocks["streamer"] = streamer
        elif isinstance(streamer, dict):
            create_kwargs["streamer_kwargs"] = streamer
            
        if isinstance(remapper, BaseBlock):
            self.blocks["remapper"] = remapper
        elif isinstance(remapper, dict):
            create_kwargs["remapper_kwargs"] = remapper
            
        if isinstance(preprocessing, BaseBlock):
            self.blocks["preprocessing"] = preprocessing
        elif isinstance(preprocessing, dict):
            create_kwargs["preprocessing_kwargs"] = preprocessing
            
        if isinstance(standardizer, BaseBlock):
            self.blocks["standardizer"] = standardizer
        elif isinstance(standardizer, dict):
            create_kwargs["standardizer_kwargs"] = standardizer
        
        if isinstance(time_buffer, BaseBlock):
            self.blocks["time_buffer"] = time_buffer
        elif isinstance(time_buffer, dict):
            create_kwargs["time_buffer_kwargs"] = time_buffer
        
        if isinstance(detector, BaseBlock):
            self.blocks["detector"] = detector
        elif isinstance(detector, dict):
            create_kwargs["detector_kwargs"] = detector

        # Construct Parent #
        super().construct(*args, create_kwargs=create_kwargs, link_kwargs=link_kwargs, **kwargs)

    # Blocks
    def create_detector(self, *args, **kwargs) -> BaseBlock:
        return self.model.trainer

    def create_blocks(
        self,
        streamer_kwargs: dict[str, Any] | None = None,
        remapper_kwargs: dict[str, Any] | None = None,
        preprocessing_kwargs: dict[str, Any] | None = None,
        standardizer_kwargs: dict[str, Any] | None = None,
        time_buffer_kwargs: dict[str, Any] | None = None,
        detector_kwargs: dict[str, Any] | None = None,
        *args: Any,
        override: bool = False,
        **kwargs: Any,
    ) -> None:
        # Create Blocks
        if override or "streamer" not in self.blocks:
            self.blocks["streamer"] = self.streamer_type(**(streamer_kwargs or {}))

        if override or "remapper" not in self.blocks:
            self.blocks["remapper"] = self.remapper_type(**(remapper_kwargs or {}))

        if override or "preprocessing" not in self.blocks:
            self.blocks["preprocessing"] = self.preprocessing_type(**(preprocessing_kwargs or {}))

        if override or "standardizer" not in self.blocks:
            self.blocks["standardizer"] = self.standardizer_type(**(standardizer_kwargs or {}))

        if override or "time_buffer" not in self.blocks:
            self.blocks["time_buffer"] = self.time_buffer_type(**(time_buffer_kwargs or {}))

        if override or "detector" not in self.blocks:
            self.blocks["detector"] = self.create_detector(**(detector_kwargs or {}))

    # IO
    def link_inner_io(self, *args: Any, **kwargs: Any) -> None:
        # Get Blocks
        streamer = self.blocks["streamer"]
        remapper = self.blocks["remapper"]
        preprocessing = self.blocks["preprocessing"]
        standardizer = self.blocks["standardizer"]
        time_buffer = self.blocks["time_buffer"]
        detector = self.blocks["detector"]

        # Inner IO
        streamer.outputs.link_forward("data", remapper.inputs, "data")
        remapper.outputs.link_forward("remapped_data", preprocessing.inputs, "data")
        preprocessing.outputs.link_forward("filter_data", standardizer.inputs, "data")
        standardizer.outputs.link_forward("features", time_buffer.inputs, "data")
        time_buffer.outputs.link_forward("buffer_data", detector.inputs, "data")

        # Set Output
        self.outputs = detector.outputs  # Make an indirect assignment
