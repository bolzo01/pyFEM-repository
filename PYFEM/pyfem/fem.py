#!/usr/bin/env python
"""
Module for FEA procedures.

Created: 2025/10/08 17:11:28
Last modified: 2025/11/09 13:23:51
Author: Francesco Bolzonella (francesco.bolzonella.1@studenti.unipd.it)
"""

import numpy as np

from .dof_types import DOFSpace
from .element_properties import ElementProperties, param
from .mesh import Mesh


def assemble_global_stiffness_matrix(
    mesh: Mesh,
    element_properties: ElementProperties,
    global_stiffness_matrix: np.ndarray,
    dof_space: DOFSpace,
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
    # print("\n- Assembling local stiffness matrix into global stiffness matrix")
    for element_index in range(num_elements):
        # Generate the local stiffness matrix for a one-dimensional spring element
        label = mesh.element_property_labels[element_index]
        elem_prop = element_properties[label]

        # Generate local stiffness matrix based on element type
        if elem_prop.kind == "spring_1D":
            k_e = param(elem_prop, "k", float)

            # print(f"\n-- Element {element_index}, k = {k_e}")
            local_stiffness_matrix = np.array([[k_e, -k_e], [-k_e, k_e]])

        elif elem_prop.kind == "bar_1D":
            E = param(elem_prop, "E", float)
            A = param(elem_prop, "A", float)

            # Get element nodes and compute length
            element_nodes = element_connectivity[element_index]
            node1, node2 = element_nodes
            x1 = mesh.points[node1]
            x2 = mesh.points[node2]
            L = x2 - x1

            # Bar stiffness matrix
            k_e = (E * A) / L
            # print(f"\n-- Element {element_index}, E = {E}, A = {A}, L = {L}")
            local_stiffness_matrix = np.array([[k_e, -k_e], [-k_e, k_e]])

        elif elem_prop.kind == "bar_2D":
            E = param(elem_prop, "E", float)
            A = param(elem_prop, "A", float)

            # Get element nodes and compute length
            element_nodes = element_connectivity[element_index]
            node1, node2 = element_nodes
            P1 = mesh.points[node1]
            P2 = mesh.points[node2]

            L = np.sqrt((P2[0] - P1[0]) ** 2 + (P2[1] - P2[1]) ** 2)

            # Calculate the directional cosines of the bar_2D element
            # (cosine and sine of angle between local and global axes)
            directional_cosines = (P2 - P1) / L
            c, s = directional_cosines

            # Bar stiffness matrix
            k_e = (E * A) / L
            # print(f"\n-- Element {element_index}, E = {E}, A = {A}, L = {L}")
            local_stiffness_matrix = k_e * np.array(
                [
                    [c * c, c * s, -c * c, -c * s],
                    [c * s, s * s, -c * s, -s * s],
                    [-c * c, -c * s, c * c, c * s],
                    [-c * s, -s * s, c * s, s * s],
                ]
            )

        else:
            raise ValueError(f"Unknown element kind: {elem_prop.kind}")

        # print("   Local K:\n  ", local_stiffness_matrix)

        # Map local degrees of freedom to global degrees of freedom for an element
        # first determine the element nodes through the element connectivity matrix
        element_nodes = element_connectivity[element_index]
        dof_mapping = dof_space.get_dof_mapping(element_nodes)

        # Assemble the local stiffness matrix into the global stiffness matrix
        for i, global_i in enumerate(dof_mapping):
            for j, global_j in enumerate(dof_mapping):
                global_stiffness_matrix[global_i, global_j] += local_stiffness_matrix[
                    i, j
                ]

    return None


def apply_nodal_forces(
    applied_forces: list[tuple[int, float]] | None,
    global_force_vector: np.ndarray,
) -> None:
    """
    Applies nodal forces to the global force vector (Neumann boundary conditions).

    This function modifies the global force vector in place.

    Args:
        applied_forces: List of (dof_index, force_value) pairs.
        global_force_vector: The global force vector to modify.
    """
    if not applied_forces:
        # Nothing to apply (handles None or empty list)
        return

    for dof, value in applied_forces:
        global_force_vector[int(dof)] = value

    return None


def apply_prescribed_displacements(
    prescribed_displacements: list[tuple[int, float]],
    global_stiffness_matrix: np.ndarray,
    global_force_vector: np.ndarray,
) -> None:
    """
    Applies prescribed displacements (Dirichlet BCs) by modifying
    the global stiffness matrix and force vector in place.

    Args:
        prescribed_displacements: List of (global_dof, value) pairs.
        global_stiffness_matrix: Global stiffness matrix (modified in place).
        global_force_vector: Global force vector (modified in place).
    """

    # Step 1: Extract DOF indices and values
    dof_indices = [int(dof) for dof, _ in prescribed_displacements]
    values = np.array([value for _, value in prescribed_displacements], dtype=float)

    # Step 2: Modify RHS -> equivalent force adjustment
    global_force_vector[:] -= global_stiffness_matrix[:, dof_indices] @ values

    # Step 3: Zero out corresponding rows and columns
    for dof, value in prescribed_displacements:
        global_stiffness_matrix[:, dof] = 0.0
        global_stiffness_matrix[dof, :] = 0.0
        global_stiffness_matrix[dof, dof] = 1.0
        global_force_vector[dof] = value  # Enforce displacement value

    return None
