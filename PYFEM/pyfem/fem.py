#!/usr/bin/env python
"""
Module for FEA procedures.

Created: 2025/10/08 17:11:28
Last modified: 2025/10/12 21:51:27
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from .mesh import Mesh


def assemble_global_stiffness_matrix(
    mesh: Mesh,
    element_stiffness: list[float],
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
        msg = (
            f"\n-- Generating stiffness matrix for element {element_index}"
            f" with stiffness {element_stiffness[element_index]}"
        )
        print(msg)

        # Generate the local stiffness matrix for a one-dimensional spring element
        stiffness_value = element_stiffness[element_index]
        local_stiffness_matrix = np.array(
            [
                [stiffness_value, -stiffness_value],
                [-stiffness_value, stiffness_value],
            ]
        )
        print(local_stiffness_matrix)

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


def compute_strain_energy_local(
    mesh: Mesh,
    element_stiffness: list[float],
    nodal_displacements: np.ndarray,
) -> None:
    """
    Computes the total strain energy by summing the strain energy of each element.

    Returns:
        None.
    """

    num_elements = mesh.num_elements
    element_connectivity = mesh.element_connectivity

    total_strain_energy = 0.0
    for element_index in range(num_elements):
        k_spring = element_stiffness[element_index]
        node1, node2 = element_connectivity[element_index]
        u1 = nodal_displacements[node1]
        u2 = nodal_displacements[node2]
        delta = u2 - u1
        strain_energy = 0.5 * k_spring * delta**2
        total_strain_energy += strain_energy
        print(f"\n- Strain energy in element {element_index}: {strain_energy}")
    print(
        f"\n- Total strain energy in the system (from local computation): {total_strain_energy}"
    )


def compute_strain_energy_global(
    original_global_stiffness_matrix: np.ndarray,
    nodal_displacements: np.ndarray,
) -> None:
    """
    Computes the total strain energy using the global solution (U = 0.5 * u^T * K * u).

    Returns:
        None.
    """

    K = original_global_stiffness_matrix
    u = nodal_displacements
    total_strain_energy = 0.5 * (u.T @ (K @ u))

    print(
        f"\n- Total strain energy in the system (from global computation): {total_strain_energy}"
    )
