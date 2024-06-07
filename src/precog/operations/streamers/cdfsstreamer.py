""" baseoperation.py

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
from typing import Any

# Third-Party Packages #
from cdfs import CDFS

# Local Packages #
from .proxyarraystreamer import ProxyArrayStreamer


# Definitions #
# Classes #
class CDFSStreamer(ProxyArrayStreamer):
    default_create_generator: str = "create_islice_time"

    cdfs: CDFS | None = None

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        cdfs: CDFS | None = None,
        empty_signal: Any = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #

        # Parent Attributes #
        super().__init__(*args, init=False, **kwargs)

        # Construct #
        if init:
            self.construct(
                cdfs=cdfs,
                empty_signal=empty_signal,
                *args,
                **kwargs,
            )

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        cdfs: CDFS | None = None,
        empty_signal: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            *args: Arguments for inheritance.
            **kwargs: Keyword arguments for inheritance.
        """
        proxy_array = None

        if cdfs is not None:
            self.cdfs = cdfs
            if cdfs.data is None:
                cdfs.open(mode="r", load=True)
            proxy_array = cdfs.data

        # Construct Parent #
        super().construct(
                proxy_array=proxy_array,
                empty_signal=empty_signal,
                *args,
                **kwargs,
            )

