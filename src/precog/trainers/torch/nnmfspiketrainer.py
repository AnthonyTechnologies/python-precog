"""nnmfspiketrainer.py

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
from collections.abc import Mapping
from typing import ClassVar, Any

# Third-Party Packages #
from blockobjects import BaseBlock, BlockGroup
from blockobjects.io import DelegatingIOManager, BaseIO, IORouter

# Local Packages #
from ...architectures.torch import BaseNNMFModule, NNMFDModule
from ...basis import ModelBasis
from ...basis.modifiers import AdaptiveMultiplicativeModifier
from ...basis.modifiers.operations import AdaptiveMultiplicativeOperation
from ..bases import BaseTrainerOperation


# Definitions #
# Classes #
class NNMFSpikeTrainer(BlockGroup, BaseTrainerOperation):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names:  ClassVar[tuple[str, ...]] = ("bases",)

    # Attributes #
    W_modifier_type: type = AdaptiveMultiplicativeOperation
    H_modifier_type: type = AdaptiveMultiplicativeOperation
    W_refiner_type: type = None
    H_refiner_type: type = None
    architecture_type: type = NNMFDModule
    
    W_modifier_kwargs: dict[str, Any]
    H_modifier_kwargs: dict[str, Any]
    W_refiner_kwargs: dict[str, Any]
    H_refiner_kwargs: dict[str, Any]
    _W_architecture: BaseNNMFModule | None = None
    _H_architecture: BaseNNMFModule | None = None

    # Properties #
    @property
    def W_architecture(self) -> BaseNNMFModule | None:
        if (op := self.operations.get("W_modifier", None)) is not None:
            return op.module
        else:
            return self._W_architecture
        
    @W_architecture.setter
    def W_architecture(self, value: BaseNNMFModule | None) -> None:
        self._W_architecture = value
        if (op := self.operations.get("W_modifier", None)) is not None:
            op.modifier.module = value

    @property
    def H_architecture(self) -> BaseNNMFModule | None:
        if (op := self.operations.get("H_modifier", None)) is not None:
            return op.module
        else:
            return self._H_architecture

    @H_architecture.setter
    def H_architecture(self, value: BaseNNMFModule | None) -> None:
        self._H_architecture = value
        if (op := self.operations.get("H_modifier", None)) is not None:
            op.modifier.module = value

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        W_modifier: BaseBlock | BaseNNMFModule | AdaptiveMultiplicativeModifier | dict[str, Any] | None = None,
        H_modifier: BaseBlock | BaseNNMFModule | AdaptiveMultiplicativeModifier | dict[str, Any] | None = None,
        W_refiner: BaseBlock | dict[str, Any] | None = None,
        H_refiner: BaseBlock | dict[str, Any] | None = None,
        *args: Any,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        subtrainers: dict[str, BaseTrainerOperation] | None = None,
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
        # New Attributes #
        self.W_modifier_kwargs = {}
        self.H_modifier_kwargs = {}
        self.W_refiner_kwargs = {}
        self.H_refiner_kwargs = {}

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                W_modifier=W_modifier,
                H_modifier=H_modifier,
                W_refiner=W_refiner,
                H_refiner=H_refiner,
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

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        W_modifier: BaseBlock | BaseNNMFModule | AdaptiveMultiplicativeModifier | dict[str, Any] | None = None,
        H_modifier: BaseBlock | BaseNNMFModule | AdaptiveMultiplicativeModifier | dict[str, Any] | None = None,
        W_refiner: BaseBlock | dict[str, Any] | None = None,
        H_refiner: BaseBlock | dict[str, Any] | None = None,
        *args: Any,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        subtrainers: dict[str, BaseTrainerOperation] | None = None,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        subtrainers_kwargs: dict[str, dict[str, Any]] | None = None,
        operations: Mapping[str, BaseBlock] | None = None,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: bool = None,
        **kwargs: Any,
    ) -> None:
        # New Setup #
        if state_variables is not None:
            if (wm_sv := state_variables.get("w_modifier", None)) is not None:
                if (_wm_sv := self.W_modifier_kwargs.get("state_variables", None)) is None:
                    _wm_sv = self.W_modifier_kwargs["state_variables"] = {}
                _wm_sv.update(wm_sv)

            if (wr_sv := state_variables.get("w_refiner", None)) is not None:
                if (_wr_sv := self.W_refiner_kwargs.get("state_variables", None)) is None:
                    _wr_sv = self.W_refiner_kwargs["state_variables"] = {}
                _wr_sv.update(wr_sv)

            if (hm_sv := state_variables.get("h_modifier", None)) is not None:
                if (_hm_sv := self.H_modifier_kwargs.get("state_variables", None)) is None:
                    _hm_sv = self.H_modifier_kwargs["state_variables"] = {}
                _hm_sv.update(hm_sv)

            if (hr_sv := state_variables.get("h_refiner", None)) is not None:
                if (_hr_sv := self.H_refiner_kwargs.get("state_variables", None)) is None:
                    _hr_sv = self.H_refiner_kwargs["state_variables"] = {}
                _hr_sv.update(hr_sv)

            state_variables = state_variables.get("local", None)

        if operations is None:
            operations = {}
        
        if isinstance(W_modifier, BaseBlock):
            operations["W_modifier"] = W_modifier
        elif isinstance(W_modifier, BaseNNMFModule):
            self.W_architecture = W_modifier
        elif isinstance(W_modifier, AdaptiveMultiplicativeModifier):
            self.W_modifier_kwargs.update(modifier=W_modifier)
        elif isinstance(W_modifier, dict):
            self.W_modifier_kwargs.update(W_modifier)
            
        if isinstance(H_modifier, BaseBlock):
            operations["H_modifier"] = H_modifier
        elif isinstance(H_modifier, BaseNNMFModule):
            self.H_architecture = H_modifier
        elif isinstance(H_modifier, AdaptiveMultiplicativeModifier):
            self.H_modifier_kwargs.update(modifier=H_modifier)
        elif isinstance(H_modifier, dict):
            self.H_modifier_kwargs.update(H_modifier)
            
        if isinstance(W_refiner, BaseBlock):
            operations["W_refiner"] = W_refiner
        elif isinstance(W_refiner, dict):
            self.W_refiner_kwargs.update(W_refiner)
            
        if isinstance(H_refiner, BaseBlock):
            operations["H_refiner"] = H_refiner
        elif isinstance(H_refiner, dict):
            self.H_refiner_kwargs.update(H_refiner)

        if not operations:
            operations = None

        # Construct Parent #
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

    # State Variables
    def get_state_variables(self) -> dict[str, Any]:
        state_vars = super().get_state_variables()
        state_vars.update({
            "W_modifer": {} if (W := self.operations.get("W_modifier", None)) is None else W.state_variables,
            "H_modifer": {} if (H := self.operations.get("H_modifier", None)) is None else H.state_variables,
            "W_refiner": {} if (w := self.operations.get("W_refiner", None)) is None else w.state_variables,
            "H_refiner": {} if (h := self.operations.get("H_refiner", None)) is None else h.state_variables,
        })
        return state_vars

    # Modifiers
    def create_W_modifier_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        self.W_modifier_kwargs.update(kwargs)
        
        if "modifier" not in self.W_modifier_kwargs:
            self.W_modifier_kwargs["modifier"] = AdaptiveMultiplicativeModifier(
                bases=self.W_modifier_kwargs.get("bases", None),
                state_variables=self.W_modifier_kwargs.get("state_variables", None),
                module=self.W_architecture or self.architecture_type(),
                updating_basis_name="W",
            )

        return self.W_modifier_kwargs

    def create_H_modifier_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        self.H_modifier_kwargs.update(kwargs)

        if "modifier" not in self.H_modifier_kwargs:
            self.H_modifier_kwargs["modifier"] = AdaptiveMultiplicativeModifier(
                bases=self.H_modifier_kwargs.get("bases", None),
                state_variables=self.H_modifier_kwargs.get("state_variables", None),
                module=self.H_architecture or self.architecture_type(),
                updating_basis_name="H",
            )

        return self.H_modifier_kwargs

    # Operations
    def create_operations(
        self,
        H_modifier_kwargs: dict[str, Any] | None = None,
        H_refiner_kwargs: dict[str, Any] | None = None,
        W_modifier_kwargs: dict[str, Any] | None = None,
        W_refiner_kwargs: dict[str, Any] | None = None,
        *args: Any,
        override: bool = False,
        **kwargs: Any,
    ) -> None:
        # Create Operations
        if "H_modifier" not in self.operations or override:
            self.operations["H_modifier"] = self.H_modifier_type(**(H_modifier_kwargs or {}))

        if self.H_refiner_type is not None and ("H_refiner" not in self.operations or override):
            self.operations["H_refiner"] = self.H_refiner_type(**(H_refiner_kwargs or {}))

        if "W_modifier" not in self.operations or override:
            self.operations["W_modifier"] = self.W_modifier_type(**(W_modifier_kwargs or {}))

        if self.W_refiner_type is not None and ("W_refiner" not in self.operations or override):
            self.operations["W_refiner"] = self.W_refiner_type(**(W_refiner_kwargs or {}))

    # IO
    def link_inner_io(self, *args: Any, **kwargs: Any) -> None:
        # Get Operations
        H_modifier = self.operations["H_modifier"]
        H_refiner = self.operations.get("H_refiner", None)

        W_modifier = self.operations["W_modifier"]
        W_refiner = self.operations.get("W_refiner", None)

        # Set Input
        self.inputs["data"] = H_modifier.inputs["data"]

        # Inner IO
        if H_refiner is not None:
            H_modifier.outputs["m_bases"] = bases_separator_H = DelegatingIOManager(names=("H",))
            bases_separator_H["H"] = H_refiner.inputs["basis"]
            H_refiner.outputs["r_bases"] = W_modifier.inputs["bases"]
        else:
            H_modifier.outputs["m_bases"] = W_modifier.inputs["bases"]

        if W_refiner is not None:
            W_modifier.outputs["m_bases"] = bases_separator_W = DelegatingIOManager(names=("W",))
            bases_separator_W["W"] = W_refiner.inputs["bases"]

            # Set Output
            W_refiner.outputs["r_bases"] = self.outputs["bases"]
        else:
            # Set Output
            W_modifier.outputs["m_bases"] = self.outputs["bases"]

    # Setup
    def setup(
        self,
        *args: Any,
        create: bool = True,
        create_kwargs: dict[str, Any] | None = None,
        link: bool = True,
        link_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Creates the inner blocks and links their IO.

        Args:
            *args: The arguments for setup.
            create: Determines if the inner operation will be created.
            create_kwargs: The keyword arguments for creating the inner blocks.
            link: Determines if the inner IO will be linked between blocks.
            link_kwargs: The keyword arguments for creating linking the inner blocks' IO.
            **kwargs: The keyword arguments for setup.
        """
        if create_kwargs is None:
            create_kwargs = {}

        if "W_modifier" not in self.operations:
            create_kwargs["W_modifier_kwargs"] = self.create_W_modifier_kwargs(
                **(create_kwargs.get("W_modifier_kwargs", {})),
            )
            
        if "H_modifier" not in self.operations:
            create_kwargs["H_modifier_kwargs"] = self.create_H_modifier_kwargs(
                **(create_kwargs.get("H_modifier_kwargs", {})),
            )
        
        super().setup(
            *args, 
            create=create,
            create_kwargs=create_kwargs,
            link=link,
            link_kwargs=link_kwargs,
            **kwargs,
        )
