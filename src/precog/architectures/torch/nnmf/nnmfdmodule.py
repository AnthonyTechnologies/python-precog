"""nnmfdmodule.py

"""
# Package Header #
from ....header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from typing import ClassVar, Any

# Third-Party Packages #
from baseobjects.functions import CallableMultiplexer, FunctionRegister
from torch import Tensor
from torch.nn.functional import conv1d, conv2d, conv3d
from torch.nn import Parameter

# Local Packages #
from ....basis import ModelBasis
from .basennmfmodule import BaseNNMFModule


# Definitions #
class NNMFDModule(BaseNNMFModule):
    # Class Attributes #
    conv_register: ClassVar[FunctionRegister] = FunctionRegister(
        conv1d=conv1d,
        conv2d=conv2d,
        conv3d=conv3d,
    )

    # Attributes #
    _padding_size: tuple[int, ...] = ()
    _flip_axes: tuple[int, ...] = ()
    conv: CallableMultiplexer

    # Properties #
    @property
    def padding_size(self) -> tuple[int, ...]:
        if not self._padding_size:
            self.convolution_setup()
        return self._padding_size

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        W: ModelBasis | Parameter | dict[str, Any] | None = None,
        H: ModelBasis | Parameter | dict[str, Any] | None = None,
        conv_type: str | None = None,
        *args: Any,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        subarchitectures: dict[str, "BaseArchitecture"] | None = None,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        subarchitectures_kwargs: dict[str, dict[str, Any]] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Attributes #
        self.conv: CallableMultiplexer = CallableMultiplexer(register=self.conv_register)

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                W=W,
                H=H,
                conv_type=conv_type,
                bases=bases,
                state_variables=state_variables,
                subarchitectures=subarchitectures,
                create_defaults=create_defaults,
                bases_kwargs=bases_kwargs,
                subarchitectures_kwargs=subarchitectures_kwargs,
                **kwargs,
            )

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        W: ModelBasis | Parameter | dict[str, Any] | None = None,
        H: ModelBasis | Parameter | dict[str, Any] | None = None,
        conv_type: str | None = None,
        *args: Any,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        subarchitectures: dict[str, "BaseArchitecture"] | None = None,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        subarchitectures_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        # Construct Parent #
        super().construct(
            W=W,
            H=H,
            bases=bases,
            state_variables=state_variables,
            subarchitectures=subarchitectures,
            create_defaults=create_defaults,
            bases_kwargs=bases_kwargs,
            subarchitectures_kwargs=subarchitectures_kwargs,
            **kwargs,
        )

        # Construct New #
        if conv_type is not None:
            self.conv.select(conv_type)
        elif self.W is not None and self.W.tensor is not None:
            self.convolution_setup()

    def convolution_setup(self):
        shape = self.W.tensor.shape
        ndim = len(shape)
        self._padding_size = tuple(shape[d] - 1 for d in range(2, len(shape)))
        if ndim == 3:
            self.conv.select("conv1d")
            self._flip_axes = (2,)
        elif ndim == 4:
            self.conv.select("conv2d")
            self._flip_axes = (2, 3)
        elif ndim == 5:
            self.conv.select("conv3d")
            self._flip_axes = (2, 3, 4)

    # Instance Methods #
    def reconstruct(self, *args, **kwargs) -> Tensor:
        """Creates a reconstruction by taking the product of W and H."""
        return self.conv(self.H.tensor, self.W.tensor.flip(self._flip_axes), padding=self.padding_size)
