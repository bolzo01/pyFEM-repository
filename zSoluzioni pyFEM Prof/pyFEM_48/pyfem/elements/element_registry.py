#!/usr/bin/env python
"""
Registry for finite-element classes and factory helpers.

Created: 2025/11/15 19:33:25
Last modified: 2025/11/17 22:15:57
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from ..element_properties import ElementProperty

ELEMENT_REGISTRY = {}

# Elements that require a material object to compute stiffness/stress
ELEMENTS_THAT_REQUIRE_MATERIAL = {
    "bar_1D",
    "bar3_1D",
    "bar_2D",
    # "spring_1D", # doesn't require material properties (k is a parameter)
}


def register_element(kind):
    def decorator(cls):
        ELEMENT_REGISTRY[kind] = cls
        return cls

    return decorator


def create_element(elem_prop: ElementProperty):
    kind = elem_prop.kind
    try:
        cls = ELEMENT_REGISTRY[kind]
    except KeyError:
        known = ", ".join(ELEMENT_REGISTRY.keys())
        raise ValueError(f"Unknown element kind '{kind}'.\n Known kinds: {known}")
    return cls(elem_prop.params, elem_prop.meta)
