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
from blockobjects.io import BaseIO, IORouter

# Local Packages #
from ...architectures.torch import BaseNNMFModule, NNMFDModule
from ...basis import ModelBasis
from ...basis.modifiers import AdaptiveMultiplicativeModifier
from ...basis.modifiers.blocks import AdaptiveMultiplicativeBlock
from ..bases import BaseTrainerBlock


# Definitions #
# Classes #
class NNMFSpikeTrainer(BlockGroup, BaseTrainerBlock):
    # Class Attributes #
    default_input_names: ClassVar[tuple[str, ...]] = ("data",)
    default_output_names:  ClassVar[tuple[str, ...]] = ("bases",)

    # Attributes #
    W_modifier_type: type = AdaptiveMultiplicativeBlock
    H_modifier_type: type = AdaptiveMultiplicativeBlock
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
        if (op := self.blocks.get("W_modifier", None)) is not None:
            return op.module
        else:
            return self._W_architecture
        
    @W_architecture.setter
    def W_architecture(self, value: BaseNNMFModule | None) -> None:
        self._W_architecture = value
        if (op := self.blocks.get("W_modifier", None)) is not None:
            op.modifier.module = value

    @property
    def H_architecture(self) -> BaseNNMFModule | None:
        if (op := self.blocks.get("H_modifier", None)) is not None:
            return op.module
        else:
            return self._H_architecture

    @H_architecture.setter
    def H_architecture(self, value: BaseNNMFModule | None) -> None:
        self._H_architecture = value
        if (op := self.blocks.get("H_modifier", None)) is not None:
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
        subtrainers: dict[str, BaseTrainerBlock] | None = None,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        subtrainers_kwargs: dict[str, dict[str, Any]] | None = None,
        blocks: Mapping[str, BaseBlock] | None = None,
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
                blocks=blocks,
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
        subtrainers: dict[str, BaseTrainerBlock] | None = None,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        subtrainers_kwargs: dict[str, dict[str, Any]] | None = None,
        blocks: Mapping[str, BaseBlock] | None = None,
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

        if blocks is None:
            blocks = {}
        
        if isinstance(W_modifier, BaseBlock):
            blocks["W_modifier"] = W_modifier
        elif isinstance(W_modifier, BaseNNMFModule):
            self.W_architecture = W_modifier
        elif isinstance(W_modifier, AdaptiveMultiplicativeModifier):
            self.W_modifier_kwargs.update(modifier=W_modifier)
        elif isinstance(W_modifier, dict):
            self.W_modifier_kwargs.update(W_modifier)
            
        if isinstance(H_modifier, BaseBlock):
            blocks["H_modifier"] = H_modifier
        elif isinstance(H_modifier, BaseNNMFModule):
            self.H_architecture = H_modifier
        elif isinstance(H_modifier, AdaptiveMultiplicativeModifier):
            self.H_modifier_kwargs.update(modifier=H_modifier)
        elif isinstance(H_modifier, dict):
            self.H_modifier_kwargs.update(H_modifier)
            
        if isinstance(W_refiner, BaseBlock):
            blocks["W_refiner"] = W_refiner
        elif isinstance(W_refiner, dict):
            self.W_refiner_kwargs.update(W_refiner)
            
        if isinstance(H_refiner, BaseBlock):
            blocks["H_refiner"] = H_refiner
        elif isinstance(H_refiner, dict):
            self.H_refiner_kwargs.update(H_refiner)

        if not blocks:
            blocks = None

        # Construct Parent #
        super().construct(
            bases=bases,
            state_variables=state_variables,
            subtrainers=subtrainers,
            create_defaults=create_defaults,
            bases_kwargs=bases_kwargs,
            subtrainers_kwargs=subtrainers_kwargs,
            blocks=blocks,
            **kwargs,
        )

    # State Variables
    def get_state_variables(self) -> dict[str, Any]:
        state_vars = super().get_state_variables()
        state_vars.update({
            "W_modifer": {} if (W := self.blocks.get("W_modifier", None)) is None else W.state_variables,
            "H_modifer": {} if (H := self.blocks.get("H_modifier", None)) is None else H.state_variables,
            "W_refiner": {} if (w := self.blocks.get("W_refiner", None)) is None else w.state_variables,
            "H_refiner": {} if (h := self.blocks.get("H_refiner", None)) is None else h.state_variables,
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

    # Blocks
    def create_blocks(
        self,
        H_modifier_kwargs: dict[str, Any] | None = None,
        H_refiner_kwargs: dict[str, Any] | None = None,
        W_modifier_kwargs: dict[str, Any] | None = None,
        W_refiner_kwargs: dict[str, Any] | None = None,
        *args: Any,
        override: bool = False,
        **kwargs: Any,
    ) -> None:
        # Create Blocks
        if override or "H_modifier" not in self.blocks:
            self.blocks["H_modifier"] = self.H_modifier_type(**(H_modifier_kwargs or {}))

        if self.H_refiner_type is not None and (override or "H_refiner" not in self.blocks):
            self.blocks["H_refiner"] = self.H_refiner_type(**(H_refiner_kwargs or {}))

        if override or "W_modifier" not in self.blocks:
            self.blocks["W_modifier"] = self.W_modifier_type(**(W_modifier_kwargs or {}))

        if self.W_refiner_type is not None and (override or "W_refiner" not in self.blocks):
            self.blocks["W_refiner"] = self.W_refiner_type(**(W_refiner_kwargs or {}))

    # IO
    def create_inner_io(self, *args: Any, **kwargs: Any) -> None:
        # Get Blocks
        H_modifier = self.blocks["H_modifier"]
        H_refiner = self.blocks.get("H_refiner", None)

        W_modifier = self.blocks["W_modifier"]
        W_refiner = self.blocks.get("W_refiner", None)

        # Inner IO
        if H_refiner is not None:
            H_modifier.outputs["m_bases"] = IORouter(names=("H",))

        if W_refiner is not None:
            W_modifier.outputs["m_bases"] = IORouter(names=("W",))

    def link_inner_io(self, *args: Any, **kwargs: Any) -> None:
        # Get Blocks
        H_modifier = self.blocks["H_modifier"]
        H_refiner = self.blocks.get("H_refiner", None)

        W_modifier = self.blocks["W_modifier"]
        W_refiner = self.blocks.get("W_refiner", None)

        # Set Input
        self.inputs.link_forward("data", H_modifier.inputs, "data")

        # Inner IO
        if H_refiner is not None:
            H_modifier.outputs.link_forward("H", H_refiner, "basis")
            H_refiner.outputs.link_forward("r_bases", self.outputs, "bases")
        else:
            H_modifier.outputs.link_forward("m_bases", W_modifier.inputs, "bases")

        if W_refiner is not None:
            W_modifier.outputs.link_forward("W", W_refiner.inputs, "bases")

            # Set Output
            W_refiner.outputs.link_forward("r_bases", self.outputs, "bases")
        else:
            # Set Output
            W_modifier.outputs.link_forward("m_bases", self.outputs, "bases")
