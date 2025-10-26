#!/usr/bin/env python
"""
Module defining the FEA solvers.

Created: 2025/26/10 18:14:50
Last modified: 2025/10/26 19:53:53
Author: Francesco Bolzonella (francesco.bolzonella.1@studenti.unipd.it)
"""

import numpy as np

from .fem import (
    apply_nodal_forces,
    apply_prescribed_displacements,
    assemble_global_stiffness_matrix,
)
from .materials import Materials
from .mesh import Mesh


def solve_linear_static(
    prescribed_displacements: list[tuple[int, float]],
    applied_forces: list[tuple[int, float]],
    materials: Materials,
    mesh: Mesh,
    dofs_per_node: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solves the linear system of equations KU=F.

    Returns:
       Solution array and global stiffness matrix before applying boundary conditions.
    """
    # - Initialize arrays

    # -- Compute total number of DOFs
    total_dofs = dofs_per_node * mesh.num_nodes

    # -- Initialize the global stiffness matrix as a square matrix of zeros
    global_stiffness_matrix = np.zeros((total_dofs, total_dofs))

    # -- Initialize the global force vector with zeros
    global_force_vector = np.zeros(total_dofs)

    # - Assemble the global stiffness matrix
    assemble_global_stiffness_matrix(mesh, materials, global_stiffness_matrix)
    print("\n- Global stiffness matrix K:")
    for row in global_stiffness_matrix:
        print(row)

    # - Save a copy of the original global stiffness matrix before applying boundary conditions
    original_global_stiffness_matrix = global_stiffness_matrix.copy()

    # - Boundary conditions: Apply forces
    apply_nodal_forces(applied_forces, global_force_vector)

    # - Boundary conditions: Constrain displacements
    apply_prescribed_displacements(
        prescribed_displacements,
        global_stiffness_matrix,
        global_force_vector,
        total_dofs,
    )

    print("\n- Modified global stiffness matrix K after applying boundary conditions:")
    for row in global_stiffness_matrix:
        print(row)

    print("\n- Global force vector F after applying boundary conditions:")
    print(global_force_vector)

    # - Solve for the nodal displacements
    nodal_displacements = np.linalg.solve(global_stiffness_matrix, global_force_vector)
    print("\n- Nodal displacements U:")
    print(nodal_displacements)

    return original_global_stiffness_matrix, nodal_displacements
