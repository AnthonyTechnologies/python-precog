""" runningshiftscaler.py

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
from copy import deepcopy
from typing import ClassVar, Any

# Third-Party Packages #
from baseobjects.functions import MethodMultiplexer
from blockobjects import BaseBlock
import numpy as np
from proxyarrays import BaseProxyArray

# Local Packages #


# Definitions #
# Constants
blank_arg = object()


# Classes #
class RunningShiftScaler(BaseBlock):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names: ClassVar[tuple[str, ...]] = ("ss_data",)

    default_burn_in: str = "burn_in_iter"
    default_shift_rescale: str = "shift_rescale_modified_zscore"
    default_forget: str = "exponential_forget"

    # New Attributes #
    axis: int = 0

    previous_mean: np.ndarray | None = None
    previous_mean_fill_value: float | int = 0

    previous_variance: np.ndarray | None = None
    previous_variance_fill_value: float | int = 10 ** -12

    threshold: int | float = 10 ** 12

    previous_count: int = 0

    burn_in_count: int = 0
    burn_in_threshold: int = 0

    _forget_factor: float | None = None

    # Properties #
    @property
    def forget_factor(self) -> float:
        return self._forget_factor

    @forget_factor.setter
    def forget_factor(self, value) -> None:
        if self.forget.selected in {None, "constant_forget", "exponential_forget"}:
            if value is None:
                self.forget.select("constant_forget")
            else:
                self.forget.select("exponential_forget")
        self._forget_factor = value

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        shift_rescale: str | None = None,
        forget_factor: float | None | object = blank_arg,
        mean: np.ndarray | None = None,
        variance: np.ndarray | None = None,
        threshold: int | float | None = None,
        burn_in: int | None = None,
        axis: int | None = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.is_burning_in: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_burn_in)
        self.forget: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_forget)
        self.shift_rescale: MethodMultiplexer = MethodMultiplexer(instance=self, select=self.default_shift_rescale)

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                shift_rescale,
                forget_factor,
                mean,
                variance,
                threshold,
                burn_in,
                axis,
                *args,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        shift_rescale: str | None = None,
        forget_factor: float | None | object = blank_arg,
        mean: np.ndarray | None = None,
        variance: np.ndarray | None = None,
        threshold: int | float | None = None,
        burn_in: int | None = None,
        axis: int | None = None,
        *args: Any,
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
        if axis is not None:
            self.axis = axis

        if shift_rescale is not None:
            self.shift_rescale.select(shift_rescale)

        if forget_factor is not blank_arg:
            self.forget_factor = forget_factor

        if mean is not None:
            self.previous_mean = mean

        if variance is not None:
            self.previous_variance = variance

        if threshold is not None:
            self.threshold = threshold

        if burn_in is not None:
            self.burn_in_threshold = burn_in

        # Construct Parent #
        super().construct(*args, **kwargs)

    def create_previous_mean(self, shape, dtype):
        self.previous_mean = np.expand_dims(np.zeros(shape, dtype), self.axis)
        if self.previous_mean_fill_value != 0:
            self.previous_mean.fill(self.previous_mean_fill_value)
        return self.previous_mean

    def create_previous_variance(self, shape, dtype):
        self.previous_variance = np.expand_dims(np.ones(shape, dtype), self.axis)
        if self.previous_variance_fill_value != 1:
            self.previous_variance.fill(self.previous_variance_fill_value)
        return self.previous_variance

    # Setup
    def setup(self, previous_sum: np.ndarray | None = None, *args: Any, **kwargs: Any) -> None:
        """A method for setting up the object before it runs operation."""
        if previous_sum is not None:
            self.previous_mean = previous_sum

    # Burn-in
    def burn_in_iter(self) -> bool:
        if self.burn_in_count < self.burn_in_threshold:
            self.burn_in_count += 1
            return True
        else:
            return False

    # Forgetting
    def constant_forget(self, index, **kwargs):
        return 1 / (self.previous_count + (index + 1))

    def exponential_forget(self, **kwargs):
        return self.forget_factor

    # Shift
    def shift_mean(self, data) -> np.ndarray:
        shifted_data = np.empty_like(data)
        slices = [slice(None)] * len(data.shape)
        for i in range(data.shape[self.axis]):
            slices[self.axis] = i
            t_slices = tuple(slices)
            shifted_data[t_slices] = data[t_slices] - self.previous_mean
            self.previous_mean = self.previous_mean + shifted_data[t_slices] * self.forget(index=i)
        self.previous_count += data.shape[-1]
        return None if self.is_burning_in() else shifted_data

    def shift_decaying_mean(self, data) -> np.ndarray:
        # https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf
        slices = [slice(None)] * len(data.shape)
        slices[self.axis] = -1
        t_slices = tuple(slices)

        t_shape = [1] * len(data.shape)
        t_shape[self.axis] = -1

        n_sample = data.shape[self.axis]
        # Calculate Weights
        weights = np.empty(n_sample)
        weights.fill(self.forget_factor)
        weights = np.power(weights, np.arange(n_sample))[::-1] * (1 - weights)

        # Calculate Decayed Means
        # a^n * x0 + a^(n-1) * (1-a) * x
        new_mean = (self.forget_factor ** n_sample) * self.previous_mean + weights.reshape(t_shape) * data
        self.previous_mean = np.expand_dims(new_mean[t_slices], self.axis)

        # Shift Data
        shifted_data = data - new_mean
        return None if self.is_burning_in() else shifted_data

    def shift_rescale_zscore(self, data) -> np.ndarray:
        scaled_data = np.empty_like(data)
        slices = [slice(None)] * len(data.shape)
        for i in range(data.shape[self.axis]):
            slices[self.axis] = i
            t_slices = tuple(slices)
            forget_factor = self.forget(index=i)
            new_data = data[t_slices]

            # Shift Data by the Previous Mean
            shifted_data = new_data - self.previous_mean

            # Calculate Z-score
            scaled_data[t_slices] = shifted_data / np.sqrt(self.previous_variance)

            # Update Mean
            self.previous_mean = (1 - forget_factor) * self.previous_mean + forget_factor * new_data

            # Update Variance
            delta_variance = shifted_data * (new_data - self.previous_mean)
            self.previous_variance = (1 - forget_factor) * self.previous_variance + forget_factor * delta_variance

        self.previous_count += data.shape[-1]
        return None if self.is_burning_in() else scaled_data

    def shift_rescale_modified_zscore(self, data) -> np.ndarray:
        scaled_data = np.empty_like(data)
        slices = [slice(None)] * len(data.shape)
        for i in range(data.shape[self.axis]):
            slices[self.axis] = i
            t_slices = tuple(slices)
            forget_factor = self.forget(index=i)
            new_data = data[t_slices]

            # Get the Deviation from the Previous Mean
            data_deviation = new_data - self.previous_mean

            # Calculate Modified Z-score
            data_std = np.sqrt(self.previous_variance)
            modified_zscore = np.tanh(data_deviation / (self.threshold * data_std))
            modified_deviation = self.threshold * modified_zscore * data_std
            modified_data = modified_deviation + self.previous_mean
            scaled_data[t_slices] = modified_zscore

            # Update Mean
            self.previous_mean = (1 - forget_factor) * self.previous_mean + forget_factor * modified_data

            # Update Variance
            delta_variance = modified_deviation * (modified_data - self.previous_mean)
            self.previous_variance = (1 - forget_factor) * self.previous_variance + forget_factor * delta_variance

        self.previous_count += data.shape[-1]
        return None if self.is_burning_in() else scaled_data

    # Evaluate
    def evaluate(self, data: np.ndarray | BaseProxyArray, *args, **kwargs: Any) -> Any:
        """An abstract method which is the evaluation of this object.

        Args:
            *args: The arguments for evaluating.
            **kwargs: The keyword arguments for evaluating.

        Returns:
            The result of the evaluation.
        """
        # Input
        if self.previous_mean is None:
            self.previous_mean = data.mean(self.axis, keepdims=True)
        if self.previous_variance is None:
            if data.shape[self.axis] > 1:
                self.previous_variance = data.var(self.axis, keepdims=True, mean=self.previous_mean)
            else:
                shape = list(data.shape)
                shape.pop(self.axis)
                self.create_previous_variance(shape, data.dtype)

        # Shift Scaling
        ss_data = self.shift_rescale(data)

        # Output
        if ss_data is None:
            return None
        elif isinstance(data, BaseProxyArray):
            data_deep = data.dataless_proxy_leaf_copy()
            data_deep.data = ss_data
            return data_deep
        else:
            return ss_data
