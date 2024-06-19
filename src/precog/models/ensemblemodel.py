""" ensemblemodel.py.py

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
from typing import ClassVar, Any

# Third-Party Packages #

# Local Packages #
from ..trainers import EnsembleTrainer
from .bases import BaseModel


# Definitions #
class EnsembleModel(BaseModel):
    # Class Attributes #
    default_trainer: ClassVar[tuple[type, dict[str]]] = (EnsembleTrainer, {})

    # Instance Methods #
    # Architecture
    def build_architecture(self, *args: Any, **kwargs: Any) -> None:
        pass

    # Trainer
    def build_trainer(self, *args: Any, **kwargs: Any) -> None:
        self.trainer.subtrainers.update({n: s.trainer for n, s in self.submodels.items()})
        self.trainer.setup(*args, **kwargs)
