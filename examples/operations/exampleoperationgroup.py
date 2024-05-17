""" exampleoperationgroup.py
An Operation which generates a random array then finds the sum of the array twice and compares the outputs
"""
# Package Header #
from precog.header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from typing import Any

# Third-Party Packages #

# Local Packages #
from precog.operations.operation import OperationGroup
from precog.operations.operation.io import IORouter
from .ioappender import IOAppender
from .rngoperation import RNGOperation
from .sumoperation import SumOperation
from .isequaloperation import IsEqualOperation


# Definitions #
# Classes #
class ExampleOperationGroup(OperationGroup):
    """An Operation which generates a random array then finds the sum of the array twice and compares the outputs."""
    default_output_names: ClassVar[tuple[str, ...]] = ("group_result",)

    # Instance Methods #
    # Setup
    def setup(self, shape: tuple[int, ...] = (100, 100), *args: Any, **kwargs: Any) -> None:
        # Create Operations
        self.operations["generator"] = generator = RNGOperation(shape=shape)
        self.operations["sum_1"] = sum_1 = SumOperation()
        self.operations["sum_2"] = sum_2 = SumOperation()
        self.operations["checker"] = checker = IsEqualOperation(equals_method="unique")

        # Group Inputs
        self.inputs = generator.inputs  # Can completely replace group inputs to reroute all inputs.

        # Inner Operation IO
        # Route the RNG output to inputs of both the sum operation.
        generator_router = IORouter({"sum_1": sum_1.inputs["data"], "sum_2": sum_2.inputs["data"]})
        generator.outputs["out_array"] = generator_router

        # Aggregate the sum outputs to a single array and send to the checker.
        checker.inputs["data"] = appender = IOAppender()
        sum_1.outputs["out_number"] = appender
        sum_2.outputs["out_number"] = appender

        # Group Outputs
        checker.outputs["result"] = self.outputs["group_result"]  # Can link the IO of single IO items.

