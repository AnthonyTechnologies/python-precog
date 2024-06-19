"""basisrefinergroup.py

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
from collections.abc import Mapping
from typing import ClassVar, Any

# Third-Party Packages #

# Local Packages #
from ....operations import BaseOperation, OperationGroup
from ....operations.operation.io import IOManager
from ....basis import ModelBasis
from ....basis.modifiers import AdaptiveMultiplicativeModifier
from ....basis.modifiers.blocks import AdaptiveMultiplicativeBlock
from .basisrefiner import BasisRefiner


# Definitions #
# Classes #
class NNMFSpikeTrainer(OperationGroup, BasisRefiner):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("bases",)
    default_output_names:  ClassVar[tuple[str, ...]] = ("m_bases",)

    # Attributes #
    sparsifier_type: type = None
    smoother_type: type = None
    standardizer_type: type = None

    # Properties #

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        *args: Any,
        operations: Mapping[str, BaseOperation] | None = None,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: bool = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                bases=bases,
                state_variables=state_variables,
                create_defaults=create_defaults,
                bases_kwargs=bases_kwargs,
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
        *args: Any,
        operations: Mapping[str, BaseOperation] | None = None,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: bool = None,
        **kwargs: Any,
    ) -> None:
        # New Setup #

        # Construct Parent #
        super().construct(
            bases=bases,
            state_variables=state_variables,
            create_defaults=create_defaults,
            bases_kwargs=bases_kwargs,
            operations=operations,
            init_io=init_io,
            sets_up=sets_up,
            setup_kwargs=setup_kwargs,
            **kwargs,
        )

    # State Variables
    def get_state_variables(self) -> dict[str, Any]:
        state_vars = super().get_state_variables()
        state_vars.update({})
        return state_vars

    # Operations
    def create_operations(
        self,
        sparsifier_kwargs: dict[str, Any] | None = None,
        smoother_kwargs: dict[str, Any] | None = None,
        standardizer_kwargs: dict[str, Any] | None = None,
        *args: Any,
        override: bool = False,
        **kwargs: Any,
    ) -> None:
        # Create Operations
        if "sparsifier" not in self.operations or override:
            self.operations["sparsifier"] = self.sparsifier_type(**(sparsifier_kwargs or {}))

        if self.smoother_type is not None and ("smoother" not in self.operations or override):
            self.operations["smoother"] = self.smoother_type(**(smoother_kwargs or {}))

        if "standardizer" not in self.operations or override:
            self.operations["standardizer"] = self.standardizer_type(**(standardizer_kwargs or {}))

    # IO
    def link_inner_io(self, *args: Any, **kwargs: Any) -> None:
        # Get Operations
        sparsifier = self.operations["sparsifier"]
        smoother = self.operations.get("smoother", None)
        standardizer = self.operations["standardizer"]

        # Set Input
        self.inputs["bases"] = sparsifier.inputs["data"]

        # Inner IO

        # Set Output
        standardizer.outputs["m_bases"] = self.outputs["bases"]
