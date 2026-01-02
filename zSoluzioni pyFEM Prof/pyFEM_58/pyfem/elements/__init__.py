#!/usr/bin/env python
"""
Element base class, registry, and all concrete finite elements.

Created: 2025/11/15 19:28:51
Last modified: 2025/12/08 19:25:03
Author: Angelo Simone (angelo.simone@unipd.it)
"""

# Export the element factory (create_element) and
# import all concrete elements so that their @register_element
# decorators execute at module import time.

from .bar3_1d import Bar3_1D
from .bar_1d import Bar1D
from .bar_1d_heat import Bar1DHeat
from .bar_2d import Bar2D
from .element_registry import create_element
from .spring_1d import Spring1D
from .tetra import Tetra
from .triangle import Triangle

__all__ = [
    "Bar3_1D",
    "Bar1D",
    "Bar1DHeat",
    "Bar2D",
    "create_element",
    "Spring1D",
    "Tetra",
    "Triangle",
]
