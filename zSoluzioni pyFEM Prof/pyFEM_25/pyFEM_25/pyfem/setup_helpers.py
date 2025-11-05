#!/usr/bin/env python
"""
Helper utilities for setting up simulation spaces and problem definitions.

Created: 2025/10/25 01:23:54
Last modified: 2025/10/29 00:08:22
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from .dof_types import DOFSpace, DOFType
from .problem import Physics, Problem


def setup_dof_space_for_problem(problem: Problem, num_nodes: int) -> DOFSpace:
    """Create and configure a DOFSpace for a given Problem.

    Handles single-physics and multiphysics cases.
    """
    dof_space = DOFSpace()

    # Determine active DOFs based on the physics
    if problem.physics == Physics.MECHANICS:
        dof_types = [DOFType.U_X]
        if problem.dimension.value > 1:
            dof_types.append(DOFType.U_Y)
        if problem.dimension.value > 2:
            dof_types.append(DOFType.U_Z)

    elif problem.physics == Physics.HEAT_TRANSFER:
        dof_types = [DOFType.TEMPERATURE]

    elif problem.physics == Physics.THERMO_MECHANICS:
        # Combine mechanics and heat transfer DOFs
        dof_types = [DOFType.U_X, DOFType.TEMPERATURE]
        if problem.dimension.value > 1:
            dof_types.insert(1, DOFType.U_Y)
        if problem.dimension.value > 2:
            dof_types.insert(2, DOFType.U_Z)

    else:
        raise NotImplementedError(f"Unknown physics: {problem.physics}")

    # Activate and assign
    dof_space.activate_dof_types(*dof_types)
    dof_space.assign_dofs_to_all_nodes(num_nodes)

    return dof_space
