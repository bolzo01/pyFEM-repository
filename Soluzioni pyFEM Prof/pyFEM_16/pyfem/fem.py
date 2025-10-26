#!/usr/bin/env python
"""
Module for FEA procedures.

Created: 2025/10/08 17:11:28
Last modified: 2025/10/18 17:44:53
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from .materials import Materials, param
from .mesh import Mesh


def assemble_global_stiffness_matrix(
    mesh: Mesh,
    materials: Materials,
    global_stiffness_matrix: np.ndarray,
) -> None:
    """
    Assembles the global stiffness matrix by integrating element stiffness matrices.

    This function modifies the global stiffness matrix in place.

    Returns:
        None.
    """

    num_elements = mesh.num_elements
    element_connectivity = mesh.element_connectivity

    # Assemble the global stiffness matrix
    print("\n- Assembling local stiffness matrix into global stiffness matrix")
    for element_index in range(num_elements):
        # Generate the local stiffness matrix for a one-dimensional spring element
        label = mesh.element_material[element_index]
        mat = materials[label]
        k_e = param(mat, "k", float)
        print(f"\n-- Element {element_index}, k = {k_e}")
        local_stiffness_matrix: np.ndarray = np.array([[k_e, -k_e], [-k_e, k_e]])
        print("   Local K:\n  ", local_stiffness_matrix)

        # Map local degrees of freedom to global degrees of freedom for an element
        # first determine the element nodes through the element connectivity matrix
        element_nodes = element_connectivity[element_index]
        # then build the local to global DOF mapping
        # For elements with one DOF per node, the global DOF is the same as the node number
        dof_mapping = element_nodes

        # Assemble the local stiffness matrix into the global stiffness matrix
        for i, global_i in enumerate(dof_mapping):
            for j, global_j in enumerate(dof_mapping):
                global_stiffness_matrix[global_i, global_j] += local_stiffness_matrix[
                    i, j
                ]

    return None


def apply_nodal_forces(
    applied_forces: list[tuple[int, float]],
    global_force_vector: np.ndarray,
) -> None:
    """
    Applies nodal forces to the global force vector (Neumann boundary conditions).

    This function modifies the global force vector in place.

    Returns:
        None.
    """

    for dof, value in applied_forces:
        global_force_vector[int(dof)] = value

    return None


def apply_prescribed_displacements(
    prescribed_displacements: list[tuple[int, float]],
    global_stiffness_matrix: np.ndarray,
    global_force_vector: np.ndarray,
    total_dofs: int,
) -> None:
    """
    Applies the prescribed displacements by modifying the global stiffness
    matrix and force vector (Dirichlet boundary conditions).

    This function modifies the updated global stiffness matrix and global force vector in place.

    Returns:
        None.
    """

    for dof, value in prescribed_displacements:
        for i in range(total_dofs):
            global_stiffness_matrix[i, int(dof)] = 0.0  # Zero out the column
            global_stiffness_matrix[int(dof), i] = 0.0  # Zero out the row
        global_stiffness_matrix[int(dof), int(dof)] = 1.0  # Put one in the diagonal
        global_force_vector[int(dof)] = 0.0

    return None
