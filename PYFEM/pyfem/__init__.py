#!/usr/bin/env python
"""
PyFEM package initialization.

Created: 2025/10/25 19:28:51
Last modified: 2026/01/31 15:01:58
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from .dof_types import DOFType
from .element_compatibility import (
    get_compatible_problems,
    is_element_compatible_with_problem,
)
from .element_properties import ElementProperty, make_element_properties
from .materials import (
    Diffusion1D,
    Diffusion2D,
    LinearElastic1D,
    LinearElastic2D,
    LinearElastic3D,
)
from .materials.material_registry import make_materials
from .mesh import Mesh
from .model import Model, ModelValidationError
from .post_processor import PostProcessor
from .problem import Dimension, Physics, Problem
from .solution import Solution
from .solvers import LinearStaticSolver
from .step import ModelState, ProcedureType, Step
from .vtk_export import VTKWriter

__all__ = [
    "DOFType",
    "get_compatible_problems",
    "is_element_compatible_with_problem",
    "make_element_properties",
    "ElementProperty",
    "Diffusion1D",
    "Diffusion2D",
    "LinearElastic1D",
    "LinearElastic2D",
    "LinearElastic3D",
    "make_materials",
    "Mesh",
    "Model",
    "ModelValidationError",
    "PostProcessor",
    "Dimension",
    "Physics",
    "Problem",
    "Solution",
    "LinearStaticSolver",
    "ModelState",
    "ProcedureType",
    "Step",
    "VTKWriter",
]
