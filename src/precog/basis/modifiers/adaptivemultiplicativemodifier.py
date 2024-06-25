"""basebasismodifier.py

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
from typing import ClassVar, Any, Callable, Optional

# Third-Party Packages #
import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.nn import Module

# Local Packages #
from ..bases import ModelBasis
from .basebasismodifier import BaseBasisModifier


# Definitions #
# Classes #
class AdaptiveMultiplicativeModifier(Optimizer, BaseBasisModifier):
    # Class Attributes #
    _pickle_exclude = {
        "_optimizer_step_pre_hooks",
        "_optimizer_step_post_hooks",
        "_optimizer_state_dict_pre_hooks",
        "_optimizer_state_dict_post_hooks",
        "_optimizer_load_state_dict_pre_hooks",
        "_optimizer_load_state_dict_post_hooks",
        "_zero_grad_profile_name",
        "_warned_capturable_if_run_uncaptured",
    }
    default_state_variables: ClassVar[dict[str, Any]] = {
        "theta": None,
        "beta": None,
        "penalty": None,  # Todo: Add this later
        "pos": None,  # torch.zeros_like(theta, memory_format=torch.preserve_format)
        "neg": None,
        "step": 0,
    }

    # Attributes #
    _module: Module | None = None
    updating_basis_name: str = "H"

    # Properties #
    @property
    def module(self) -> Module | None:
        return self._module

    @module.setter
    def module(self, value: Module | None) -> None:
        if value is None and len(self.all_bases.maps) > 1:
            self.all_bases.maps.pop()
        elif value is not None:
            if len(self.all_bases.maps) == 1:
                self.all_bases.maps.append(value.all_bases)
            else:
                self.all_bases.maps[1] = value.all_bases

        self._module = value

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        module: Module | None = None,
        updating_basis_name: str | None = None,
        *args: Any,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #

        # Parent Attributes #
        Optimizer.__init__(self, params=kwargs.get("params", ({"params": ()},)), defaults=kwargs.get("defaults", {}))
        BaseBasisModifier.__init__(self, *args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                bases=bases,
                state_variables=state_variables,
                module=module,
                updating_basis_name=updating_basis_name,
                create_defaults=create_defaults,
                bases_kwargs=bases_kwargs,
                **kwargs,
            )

    # Pickling
    def __getstate__(self) -> dict[str, Any]:
        """Creates a dictionary of attributes which can be used to rebuild this object

        Returns:
            A dictionary of this object's attributes.
        """
        state = self.__dict__.copy()
        for key in self._pickle_exclude:
            if key in state:
                del state[key]
        return state

    # Instance Methods  #
    # Constructors/Destructors
    def construct(
        self,
        bases: dict[str, ModelBasis] | None = None,
        state_variables: dict[str, Any] | None = None,
        module: Module | None = None,
        updating_basis_name: str | None = None,
        *args: Any,
        create_defaults: bool = False,
        bases_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        # Assign New #
        if module is not None:
            self.module = module

        if updating_basis_name is not None:
            self.updating_basis_name = updating_basis_name

        # Construct Parent #
        super().construct(
                bases=bases,
                state_variables=state_variables,
                create_defaults=create_defaults,
                bases_kwargs=bases_kwargs,
                **kwargs,
            )

    @torch.enable_grad()
    def closure(self):
        self.zero_grad()
        return self.module.forward()

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None,
        x: Tensor | None = None,
        theta: Tensor | None = None,
        beta: int | None = None,
        penalty: Tensor | None = None,
        pos: Tensor | None = None,
        neg: Tensor | None = None,
        step: int | None = None,
    ) -> None:
        """

        Args:
            closure:
            x:
            theta:
            beta:
            penalty:
            pos:
            neg:
            step:
        """
        # Resolve State Variables
        if theta is None:
            theta = self.state_variables["theta"]
        else:
            self.state_variables["theta"] = theta

        if beta is None:
            beta = self.state_variables["beta"]
        else:
            self.state_variables["beta"] = beta
            
        if penalty is None:
            penalty = self.state_variables["penalty"]
        else:
            self.state_variables["penalty"] = penalty
            
        if pos is None:
            pos = self.state_variables["pos"]
        else:
            self.state_variables["pos"] = pos
            
        if neg is None:
            neg = self.state_variables["neg"]
        else:
            self.state_variables["neg"] = neg
            
        if step is None:
            step = self.state_variables["step"]

        # Select Basis Tensor to Update #
        tensor = self.all_bases[self.updating_basis_name].tensor
        # Cache the gradient status
        required_grad = tensor.requires_grad
        tensor.requires_grad = True

        # Create Optimizer #
        # Ensure the closure is always called with grad enabled
        closure = self.closure if closure is None else torch.enable_grad()(closure)

        # Create optimizer by closing the optimization loop
        module_optimizer = closure()

        # Initialize temporary gradient components #
        _neg = torch.zeros_like(tensor)
        _pos = torch.zeros_like(tensor)

        # Multiplicative update coefficients for beta-divergence
        #      Marmin, A., Goulart, J.H.D.M. and Févotte, C., 2021.
        #      Joint Majorization-Minimization for Nonnegative Matrix
        #      Factorization with the $\beta $-divergence.
        #      arXiv preprint arXiv:2106.15214.
        match beta:
            case 0:
                optimizer_eps = module_optimizer.add(self.precision)
                output_pos = optimizer_eps.reciprocal_()
                output_neg = output_pos.square().mul_(x)
            case 1:
                optimizer_eps = module_optimizer.add(self.precision)
                output_pos = torch.ones_like(module_optimizer)
                output_neg = np.expand_dims(np.moveaxis(x, -1, 0), 0) / optimizer_eps.detach().numpy()
            case 2:
                output_pos = module_optimizer
                output_neg = np.expand_dims(np.moveaxis(x, -1, 0), 0)
            case _:
                optimizer_eps = module_optimizer.add(self.precision)
                output_pos = optimizer_eps.pow_(beta - 1)
                output_neg = optimizer_eps.pow(beta - 2).mul_(x)

        # Generate Numerator (Negative Factor) #
        module_optimizer.backward(torch.from_numpy(output_neg), retain_graph=True)  # Optimize module to data, retain original unoptimized bases
        __neg = torch.clone(tensor.grad).relu_()  # Get gradient between unoptimized and optimized target basis
        tensor.grad.zero_()  # Reset optimizer bases

        # Generator Denominator (positive factor) #
        # The parameter gradient holds both components (positive - negative)
        module_optimizer.backward(output_pos)
        __pos = torch.clone(tensor.grad).relu_()  # Get gradient between unoptimized and optimized target basis
        tensor.grad.add_(-__neg)
        # Apply Penalty (Regulation)
        __pos.add_(penalty)

        # Add to the running estimate
        _neg.add_(__neg)
        _pos.add_(__pos)

        # Initialize the state variables for gradient averaging.
        # Enables incremental learning.

        # Accumulate gradients
        neg.mul_(1 - theta).add_(_neg.mul_(theta))
        pos.mul_(1 - theta).add_(_pos.mul_(theta))

        # Prevent zero-valued multipliers
        neg.add_(self.precision)
        pos.add_(self.precision)

        # Multiplicative Update
        multiplier = neg.div(pos)
        tensor.mul_(multiplier)

        # Force the gradient requirement to off
        tensor.requires_grad = False

        # Reinstate the grad status from before
        tensor.requires_grad = required_grad
        
        # Update Step
        self.state_variables["step"] = step + 1 
        
        return None

    def modify(
        self,
        closure: Optional[Callable] = None,
        module: Module | None = None,
        x: Tensor | None = None,
        theta: Tensor | None = None,
        beta: int | None = None,
        penalty: Tensor | None = None,
        pos: Tensor | None = None,
        neg: Tensor | None = None,
        step: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        temp_module = self.module
        if module is not None:
            self.module = module
        self.step(
            closure=closure,
            x=x,
            theta=theta,
            beta=beta,
            penalty=penalty,
            pos=pos,
            neg=neg,
            step=step,
            *args,
            **kwargs,
        )
        self.module = temp_module

    def update(
        self,
        closure: Optional[Callable] = None,
        x: Tensor | None = None,
        theta: Tensor | None = None,
        beta: int | None = None,
        penalty: Tensor | None = None,
        pos: Tensor | None = None,
        neg: Tensor | None = None,
        step: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.step(
            closure=closure,
            x=x,
            theta=theta,
            beta=beta,
            penalty=penalty,
            pos=pos,
            neg=neg,
            step=step,
            *args,
            **kwargs,
        )

        return dict(self.all_bases)
