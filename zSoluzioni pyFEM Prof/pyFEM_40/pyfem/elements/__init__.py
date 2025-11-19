#!/usr/bin/env python
"""
Element base class, registry, and all concrete finite elements.

Created: 2025/11/15 19:28:51
Last modified: 2025/11/16 22:00:21
Author: Angelo Simone (angelo.simone@unipd.it)
"""

# Export the element factory
from .element_registry import create_element

# Import all concrete elements so that their @register_element
# decorators execute at module import time.
from .spring_1d import Spring1D

__all__ = [
    "create_element",
    "Spring1D",
]
