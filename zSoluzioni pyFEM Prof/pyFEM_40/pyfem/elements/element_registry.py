"""
Registry for finite-element classes and factory helpers.

Created: 2025/11/15 19:33:25
Last modified: 2025/11/16 22:22:49
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from ..element_properties import ElementProperty

ELEMENT_REGISTRY = {}


def register_element(kind):
    def decorator(cls):
        ELEMENT_REGISTRY[kind] = cls
        return cls

    return decorator


def create_element(elem_prop: ElementProperty):
    kind = elem_prop.kind
    cls = ELEMENT_REGISTRY[kind]
    return cls(elem_prop.params, elem_prop.meta)
