"""
Material models for constitutive behavior.

Created: 2025/11/17 00:16:39
Last modified: 2025/12/08 15:22:08
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from .linear_1d import LinearElastic1D
from .linear_2d import LinearElastic2D
from .linear_3d import LinearElastic3D
from .material import Diffusion1D, Material
from .material_registry import MaterialProperties, make_materials

__all__ = [
    "Material",
    "LinearElastic1D",
    "LinearElastic2D",
    "LinearElastic3D",
    "Diffusion1D",
    "make_materials",
    "MaterialProperties",
]
