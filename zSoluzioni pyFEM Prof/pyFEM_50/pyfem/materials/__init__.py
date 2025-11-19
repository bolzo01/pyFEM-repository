"""
Material models for constitutive behavior.

Created: 2025/11/17 00:16:39
Last modified: 2025/11/17 21:30:57
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from .linear_1d import LinearElastic1D
from .material import Material
from .material_registry import MaterialProperties, make_materials

__all__ = [
    "Material",
    "LinearElastic1D",
    "make_materials",
    "MaterialProperties",
]
