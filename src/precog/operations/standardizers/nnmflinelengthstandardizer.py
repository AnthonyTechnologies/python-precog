""" nnmflinelengthstandardizer.py
An abstract class which defines an Block, an easily definable data processing block with inputs and outputs.
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
from typing import ClassVar, Any, Callable

# Third-Party Packages #
from blockobjects import BlockGroup
import numpy as np

# Local Packages #
from ..features import LineLength
from ..shiftrescalers import RunningShiftScaler, blank_arg
from ..constraints import NonNegative


# Definitions #
# Classes #
class NNMFLineLengthStandardizer(BlockGroup):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names: ClassVar[tuple[str, ...]] = ("features",)

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        squared_estimator: bool | None = None,
        window_len: int | None = None,
        window_type: Callable | None = None,
        shift_scale: str | None = None,
        forget_factor: float | None | object = blank_arg,
        mean: np.ndarray | None = None,
        variance: np.ndarray | None = None,
        threshold: int | float | None = None,
        burn_in: int | None = None,
        non_negative: str | None = None,
        non_negative_kwargs: dict[str, Any] | None = None,
        axis: int | None = None,
        *args: Any,
        setup_kwargs: dict[str, Any] | None = None,
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
                squared_estimator=squared_estimator,
                window_len=window_len,
                window_type=window_type,
                shift_scale=shift_scale,
                forget_factor=forget_factor,
                mean=mean,
                variance=variance,
                threshold=threshold,
                burn_in=burn_in,
                non_negative=non_negative,
                non_negative_kwargs=non_negative_kwargs,
                axis=axis,
                setup_kwargs=setup_kwargs,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        squared_estimator: bool | None = None,
        window_len: int | None = None,
        window_type: Callable | None = None,
        shift_scale: str | None = None,
        forget_factor: float | None | object = blank_arg,
        mean: np.ndarray | None = None,
        variance: np.ndarray | None = None,
        threshold: int | float | None = None,
        burn_in: int | None = None,
        non_negative: str | None = None,
        non_negative_kwargs: dict[str, Any] | None = None,
        axis: int | None = None,
        *args: Any,
        setup_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            sets_up: Determines if setup will run during this construction.
            setup_kwargs: The keyword arguments for the setup method.
            **kwargs: Keyword arguments for inheritance.
        """
        setup_kwargs = {} if setup_kwargs is None else setup_kwargs
        setup_kwargs["create_kwargs"] = setup_kwargs.get("create_kwargs", {}) | dict(
            squared_estimator=squared_estimator,
            window_len=window_len,
            window_type=window_type,
            shift_scale=shift_scale,
            forget_factor=forget_factor,
            mean=mean,
            variance=variance,
            threshold=threshold,
            burn_in=burn_in,
            non_negative=non_negative,
            non_negative_kwargs=non_negative_kwargs,
            axis=axis,
        )

        # Construct Parent #
        super().construct(*args, setup_kwargs=setup_kwargs, **kwargs)

    # Blocks
    def create_blocks(
        self,
        squared_estimator: bool | None = None,
        window_len: int | None = None,
        window_type: Callable | None = None,
        shift_scale: str | None = None,
        forget_factor: float | None | object = blank_arg,
        mean: np.ndarray | None = None,
        variance: np.ndarray | None = None,
        threshold: int | float | None = None,
        burn_in: int | None = None,
        non_negative: str | None = None,
        non_negative_kwargs: dict[str, Any] | None = None,
        axis: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Create Blocks
        self.blocks["line_length"] = LineLength(
            squared_estimator=squared_estimator,
            window_len=window_len,
            window_type=window_type,
            axis=axis,
        )

        self.blocks["shift_scale"] = RunningShiftScaler(
            shift_rescale=shift_scale,
            forget_factor=forget_factor,
            mean=mean,
            variance=variance,
            threshold=threshold,
            burn_in=burn_in,
            axis=axis,
        )

        self.blocks["non_negative"] = NonNegative(
            non_negative=non_negative,
            non_negative_kwargs=non_negative_kwargs,
        )

    # IO
    def link_inner_io(self, *args: Any, **kwargs: Any) -> None:
        # Get Blocks
        line_length = self.blocks["line_length"]
        shift_scaler = self.blocks["shift_scale"]
        non_negative_op = self.blocks["non_negative"]

        # Set Input
        self.inputs["data"] = line_length.inputs["data"]

        # Inner IO
        line_length.outputs["features"] = shift_scaler.inputs["data"]
        shift_scaler.outputs["ss_data"] = non_negative_op.inputs["data"]

        # Set Output
        non_negative_op.outputs["nn_data"] = self.outputs["features"]

    # Setup
    def setup(
        self,
        squared_estimator: bool | None = None,
        window_len: int | None = None,
        window_type: Callable | None = None,
        shift_scale: str | None = None,
        forget_factor: float | None | object = blank_arg,
        mean: np.ndarray | None = None,
        variance: np.ndarray | None = None,
        threshold: int | float | None = None,
        burn_in: int | None = None,
        non_negative: str | None = None,
        non_negative_kwargs: dict[str, Any] | None = None,
        axis: int | None = None,
        *args: Any,
        create: bool = True,
        create_kwargs: dict[str, Any] | None = None,
        link: bool = True,
        link_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """A method for setting up the object before it runs block."""
        new_create_kwargs = dict(
            squared_estimator=squared_estimator,
            window_len=window_len,
            window_type=window_type,
            shift_scale=shift_scale,
            forget_factor=forget_factor,
            mean=mean,
            variance=variance,
            threshold=threshold,
            burn_in=burn_in,
            non_negative=non_negative,
            non_negative_kwargs=non_negative_kwargs,
            axis=axis,
        )

        create_kwargs = new_create_kwargs | ({} if create_kwargs is None else create_kwargs)

        super().setup(create=create, create_kwargs=create_kwargs, link=link, link_kwargs=link_kwargs, **kwargs)
