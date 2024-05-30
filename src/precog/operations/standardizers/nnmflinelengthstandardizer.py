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
        create_kwargs: dict[str, Any] | None = None,
        link_kwargs: dict[str, Any] | None = None,
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
                create_kwargs=create_kwargs,
                link_kwargs=link_kwargs,
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
        create_kwargs: dict[str, Any] | None = None,
        link_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            squared_estimator: Determines if the estimator is squared.
            window_len: The length of the window.
            window_type: The type of the window.
            shift_scale: The shift scale.
            forget_factor: The forget factor.
            mean: The mean.
            variance: The variance.
            threshold: The threshold.
            burn_in: The burn-in.
            non_negative: Determines if the value is non-negative.
            non_negative_kwargs: The keyword arguments for the non-negative value.
            axis: The axis.
            *args: Additional arguments.
            create_kwargs: The keyword arguments for the create method.
            link_kwargs: The keyword arguments for the link method.
            **kwargs: Additional keyword arguments.
        """
        create_kwargs = {} if create_kwargs is None else create_kwargs

        if squared_estimator is not None:
            create_kwargs["squared_estimator"] = squared_estimator

        if window_len is not None:
            create_kwargs["window_len"] = window_len

        if window_type is not None:
            create_kwargs["window_type"] = window_type

        if shift_scale is not None:
            create_kwargs["shift_scale"] = shift_scale

        if forget_factor is not blank_arg:
            create_kwargs["forget_factor"] = forget_factor

        if mean is not None:
            create_kwargs["mean"] = mean

        if variance is not None:
            create_kwargs["variance"] = variance

        if threshold is not None:
            create_kwargs["threshold"] = threshold

        if burn_in is not None:
            create_kwargs["burn_in"] = burn_in

        if non_negative is not None:
            create_kwargs["non_negative"] = non_negative

        if non_negative_kwargs is not None:
            create_kwargs["non_negative_kwargs"] = non_negative_kwargs

        if axis is not None:
            create_kwargs["axis"] = axis

        # Construct Parent #
        super().construct(*args, create_kwargs=create_kwargs, link_kwargs=link_kwargs, **kwargs)

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
        override: bool = False,
        **kwargs: Any,
    ) -> None:
        # Create Blocks
        if override or "line_length" not in self.blocks:
            self.blocks["line_length"] = LineLength(
                squared_estimator=squared_estimator,
                window_len=window_len,
                window_type=window_type,
                axis=axis,
            )

        if override or "shift_scale" not in self.blocks:
            self.blocks["shift_scale"] = RunningShiftScaler(
                shift_rescale=shift_scale,
                forget_factor=forget_factor,
                mean=mean,
                variance=variance,
                threshold=threshold,
                burn_in=burn_in,
                axis=axis,
            )

        if override or "non_negative" not in self.blocks:
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
        self.inputs.link_forward("data", line_length.inputs, "data")

        # Inner IO
        line_length.outputs.link_forward("features", shift_scaler.inputs, "data")
        shift_scaler.outputs.link_forward("ss_data", non_negative_op.inputs, "data")

        # Set Output
        non_negative_op.outputs.link_forward("nn_data", self.outputs, "features")
