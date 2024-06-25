"""torchmodelbasis.py

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
from collections.abc import Iterable, Mapping
from typing import Any

# Third-Party Packages #
import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter

# Local Packages #
from ..bases import ModelBasis


# Definitions #
# Classes #
class TorchModelBasis(ModelBasis):
    # Attributes #
    detached_tensor_: Tensor | None = None
    ndarray_: np.ndarray | None = None

    # Properties #
    @property
    def detached_tensor(self) -> Tensor | None:
        """The detached tensor."""
        if self.detached_tensor_ is None:
            self.detached_tensor_ = self.tensor.detach()
        return self.detached_tensor_

    @property
    def ndarray(self) -> np.ndarray | None:
        """The numpy array of the tensor."""
        if self.ndarray_ is None:
            self.ndarray_ = self.detached_tensor.numpy()
        return self.ndarray_

    # Magic Methods  #
    # Construction/Destruction
    def __init__(
        self,
        tensor: Tensor | None = None,
        size: Iterable[int] | None = None,
        requires_grad: bool = True,
        state_variables: Mapping[str, Any] | None = None,
        factor_axis: int | None = None,
        *,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        component_types: dict[str, tuple[type, dict[str, Any]]] | None = None,
        components: dict[str, Any] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                tensor=tensor,
                size=size,
                requires_grad=requires_grad,
                state_variables=state_variables,
                factor_axis=factor_axis,
                component_kwargs=component_kwargs,
                component_types=component_types,
                components=components,
                **kwargs,
            )

    def construct(
        self,
        tensor: Tensor | None = None,
        size: Iterable[int] | None = None,
        requires_grad: bool = True,
        state_variables: Mapping[str, Any] | None = None,
        factor_axis: int | None = None,
        *args: Any,
        component_kwargs: dict[str, dict[str, Any]] | None = None,
        component_types: dict[str, tuple[type, dict[str, Any]]] | None = None,
        components: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if size is not None and tensor is None:
            self.create_tensor(size=size, requires_grad=requires_grad)

        # Construct Parent #
        super().construct(
            tensor=tensor,
            factor_axis=factor_axis,
            component_kwargs=component_kwargs,
            component_types=component_types,
            components=components,
            **kwargs,
        )

    def create_tensor(self, size: Iterable[int], requires_grad: bool = True, **kwargs: Any) -> Tensor:
        """Create an empty tensor which contains values.

        Args:
            size: The dimensions of the tensor.
            requires_grad:
            **kwargs: The keyword arguments for creating an empty tensor.
        """
        self.tensor = Parameter(torch.rand(*size, **kwargs), requires_grad=requires_grad)
        return self.tensor
