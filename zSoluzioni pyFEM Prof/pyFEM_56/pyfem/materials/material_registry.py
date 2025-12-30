#!/usr/bin/env python
"""
Material registry utilities.

This module defines a lightweight registry for material models used in the
finite element analysis. Materials are identified by user-defined string labels
and mapped to instances of classes derived from ``Material``.

Created: 2025/10/18 23:13:32
Last modified: 2025/11/17 22:21:45
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from .material import Material

MaterialProperties = dict[str, Material]


def make_materials(pairs) -> MaterialProperties:
    """
    Construct a material registry from a list of (label, material_instance) pairs.

    Example:
        materials = make_materials([
            ("steel", LinearElastic1D(E=200e9)),
            ("rubber", HyperElasticNeoHookean(E=10e6, nu=0.49)),
        ])
    """
    out: MaterialProperties = {}
    seen: set[str] = set()

    for idx, pair in enumerate(pairs):
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise ValueError(
                f"Item #{idx} must be a (label, material) pair, got {pair!r}"
            )

        label, material = pair

        if not isinstance(label, str) or not label:
            raise ValueError(f"Item #{idx} has invalid label {label!r}")

        if label in seen:
            raise ValueError(f"Duplicate material label '{label}'")
        seen.add(label)

        if not isinstance(material, Material):
            raise TypeError(
                f"Item #{idx}: material must be a Material instance, got {type(material)!r}"
            )

        out[label] = material

    return out
