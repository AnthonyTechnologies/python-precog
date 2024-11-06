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
from cdfs import BaseCDFS

# Local Packages #
from .proxyarraystreamer import ProxyArrayStreamer


# Definitions #
# Classes #
class CDFSStreamer(ProxyArrayStreamer):
    default_create_generator: str = "create_islice_time"

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        cdfs: BaseCDFS | None = None,
        empty_signal: Any = None,
        *args: Any,
        init: bool = True,
        **kwargs: Any,
    ) -> None:
        # New Attributes #
        self.cdfs: BaseCDFS | None = None

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
        cdfs: BaseCDFS | None = None,
        empty_signal: Any = None,
        *args: Any,
        init_io: bool = True,
        sets_up: bool = True,
        setup_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Constructs this object.

        Args:
            *args: Arguments for inheritance.
            init_io: Determines if construct_io run during this construction.
            sets_up: Determines if setup will run during this construction.
            setup_kwargs: The keyword arguments for the setup method.
            **kwargs: Keyword arguments for inheritance.
        """
        proxy_array = None

        if cdfs is not None:
            self.cdfs = cdfs
            if cdfs.components["contents"].create_contents_proxy() is None:
                cdfs.open(mode="r", load=True)
            proxy_array = cdfs.components["contents"].create_contents_proxy()

        # Construct Parent #
        super().construct(
                proxy_array=proxy_array,
                empty_signal=empty_signal,
                *args,
                **kwargs,
            )

