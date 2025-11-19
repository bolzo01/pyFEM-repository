#!/usr/bin/env python
"""
PyFEM package initialization.

Created: 2025/10/25 19:28:51
Last modified: 2025/11/16 13:02:05
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from .dof_types import DOFType
from .element_compatibility import (
    get_compatible_problems,
    is_element_compatible_with_problem,
)
from .element_properties import make_element_properties
from .mesh import Mesh
from .model import Model, ModelValidationError
from .post_processor import PostProcessor
from .problem import Dimension, Physics, Problem
from .solution import Solution
from .solvers import LinearStaticSolver
from .step import ModelState, ProcedureType, Step

__all__ = [
    "DOFType",
    "get_compatible_problems",
    "is_element_compatible_with_problem",
    "make_element_properties",
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
]
