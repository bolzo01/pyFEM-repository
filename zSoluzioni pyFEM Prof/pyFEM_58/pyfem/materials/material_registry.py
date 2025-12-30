#!/usr/bin/env python
"""
Material registry utilities.

This module defines a lightweight registry for material models used in the
finite element analysis. Materials are identified by user-defined string labels
and mapped to instances of classes derived from ``Material``.

Created: 2025/10/18 23:13:32
Last modified: 2025/12/09 01:33:42
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from .material import Material

MaterialProperties = dict[str, Material]


def make_materials(
    items: list[tuple[str, Material]],
) -> dict[str, Material]:
    """
    Construct a material registry from a list of (label, material_instance) pairs.

    Example:
        materials = make_materials([
            ("steel", LinearElastic1D(E=200e9)),
            ("soil", Diffusion1D(alpha=1.2e-6)),
        ])
    """

    out: dict[str, Material] = {}
    seen: set[str] = set()

    for idx, pair in enumerate(items):
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise ValueError(
                f"Item #{idx} must be a (label, material) pair, got {pair!r}"
            )

        label, mat = pair

        if not isinstance(label, str):
            raise ValueError(
                f"First entry in item #{idx} must be a string label, got {label!r}"
            )

        if label in seen:
            raise ValueError(f"Duplicate material label '{label}' detected")

        if not isinstance(mat, Material):
            raise TypeError(
                f"Material for label '{label}' is not a subclass of Material, "
                f"got {type(mat).__name__}"
            )

        seen.add(label)
        out[label] = mat

    return out
