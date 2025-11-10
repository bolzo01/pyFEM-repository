#!/usr/bin/env python
"""
Module for FEA procedures.

Created: 2025/10/08 17:11:28
Last modified: 2025/11/10 16:55:27
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

        # Get local to global DOF mapping using DOFSpace
        element_nodes = element_connectivity[element_index]
        dof_mapping = dof_space.get_dof_mapping(element_nodes)

        # Assemble the local stiffness matrix into the global stiffness matrix
        for i, global_i in enumerate(dof_mapping):
            for j, global_j in enumerate(dof_mapping):
                global_stiffness_matrix[global_i, global_j] += local_stiffness_matrix[
                    i, j
                ]

    return None
