#!/usr/bin/env python
"""
Element base class, registry, and all concrete finite elements.

Created: 2025/11/15 19:28:51
Last modified: 2025/11/17 01:46:43
Author: Angelo Simone (angelo.simone@unipd.it)
"""

# Export the element factory (create_element) and
# import all concrete elements so that their @register_element
# decorators execute at module import time.

from .bar_1d import Bar1D
from .bar_2d import Bar2D
from .element_registry import create_element
from .spring_1d import Spring1D

__all__ = [
    "Bar1D",
    "Bar2D",
    "create_element",
    "Spring1D",
]
