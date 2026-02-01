"""
Material models for constitutive behavior.

Created: 2025/11/17 00:16:39
Last modified: 2026/01/29 18:11:16
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from .linear_1d import LinearElastic1D
from .linear_2d import LinearElastic2D
from .linear_3d import LinearElastic3D
from .material import Diffusion1D, Diffusion2D, Material
from .material_registry import MaterialProperties, make_materials

__all__ = [
    "Material",
    "LinearElastic1D",
    "LinearElastic2D",
    "LinearElastic3D",
    "Diffusion1D",
    "Diffusion2D",
    "make_materials",
    "MaterialProperties",
]
