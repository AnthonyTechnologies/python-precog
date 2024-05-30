"""nnmfspiketrainer.py

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
from collections.abc import Mapping
from typing import ClassVar, Any

# Third-Party Packages #
from blockobjects import BaseBlock, BlockGroup
from blockobjects.io import DelegatingIOManager, BaseIO, IORouter

# Local Packages #
from ..architectures.torch import BaseNNMFModule, NNMFDModule
from ..basis import ModelBasis
from ..basis.modifiers import AdaptiveMultiplicativeModifier
from ..trainers.bases import BaseTrainerBlock


# Definitions #
# Classes #
class EnsembleTrainer(BlockGroup, BaseTrainerBlock):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names:  ClassVar[tuple[str, ...]] = ("bases",)

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        subtrainers: dict[str, BaseTrainerBlock] | None = None,
        *args: Any,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        subtrainers_kwargs: dict[str, dict[str, Any]] | None = None,
        operations: Mapping[str, BaseBlock] | None = None,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: bool = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                subtrainers=subtrainers,
                bases=bases,
                state_variables=state_variables,
                create_defaults=create_defaults,
                bases_kwargs=bases_kwargs,
                subtrainers_kwargs=subtrainers_kwargs,
                operations=operations,
                init_io=init_io,
                sets_up=sets_up,
                setup_kwargs=setup_kwargs,
                **kwargs,
            )

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        subtrainers: dict[str, BaseTrainerBlock] | None = None,
        *args: Any,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        subtrainers_kwargs: dict[str, dict[str, Any]] | None = None,
        operations: Mapping[str, BaseBlock] | None = None,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: bool = None,
        **kwargs: Any,
    ) -> None:
        # Construct Parents #
        super().construct(
            bases=bases,
            state_variables=state_variables,
            subtrainers=subtrainers,
            create_defaults=create_defaults,
            bases_kwargs=bases_kwargs,
            subtrainers_kwargs=subtrainers_kwargs,
            operations=operations,
            init_io=init_io,
            sets_up=sets_up,
            setup_kwargs=setup_kwargs,
            **kwargs,
        )

    # Operations
    def create_operations(self, *args: Any, override: bool = False, **kwargs: Any) -> None:
        """Creates the inner blocks.

        Args:
            *args: The arguments for creating the inner blocks.
            override: Determines if the inner blocks will be overridden.
            **kwargs: The keyword arguments for creating the inner blocks.
        """
        for name, trainer in self.subtrainers.items():
            if name not in self.operations or override:
                self.operations[name] = trainer

    # IO
    def link_inner_io(self, *args: Any, **kwargs: Any) -> None:
        # Get IO of Each Operation
        input_data = {}
        output_bases = {}
        for name, operation in self.operations.items():
            input_data[name] = operation.inputs["data"]
            output_bases[name] = operation.outputs["bases"]

        # Set Input
        self.inputs["data"] = IORouter(io_=input_data)

        # Set Output
        self.outputs["bases"] = IORouter(io_=output_bases)
