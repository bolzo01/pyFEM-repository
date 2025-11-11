#!/usr/bin/env python
"""
Module for FEA procedures.

Created: 2025/10/08 17:11:28
Last modified: 2025/11/08 10:00:08
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np
from scipy import sparse

from .dof_types import DOFSpace
from .element_properties import ElementProperties, ElementProperty, param
from .mesh import Mesh


def assemble_global_stiffness_matrix(
    mesh: Mesh,
    element_properties: ElementProperties,
    global_stiffness_matrix: np.ndarray,
    dof_space: DOFSpace,
    use_sparse: bool = True,
) -> np.ndarray | sparse.spmatrix:
    """
    Assembles the global stiffness matrix by integrating element stiffness matrices.

    Returns:
        The assembled global stiffness matrix (dense or sparse).
    """

    num_elements = mesh.num_elements
    element_connectivity = mesh.element_connectivity

    # ============================================================
    # SPARSE ASSEMBLY (with duplicate elimination)
    # ============================================================
    if use_sparse:
        # Use dictionary to accumulate values and eliminate duplicates
        triplet_dict: dict[tuple[int, int], float] = {}

        for element_index in range(num_elements):
            # Get element properties
            label = mesh.element_property_labels[element_index]
            elem_prop = element_properties[label]

            # Compute local stiffness matrix
            local_K = _compute_local_stiffness(
                elem_prop, element_index, element_connectivity, mesh
            )

            # Get DOF mapping
            dof_map = dof_space.get_dof_mapping(element_connectivity[element_index])

            # Accumulate into dictionary (sums duplicates manually)
            for i, i_global in enumerate(dof_map):
                for j, j_global in enumerate(dof_map):
                    key = (i_global, j_global)
                    triplet_dict[key] = triplet_dict.get(key, 0.0) + local_K[i, j]

        # Remove entries that cancel to ~0 due to numerical roundoff
        tolerance = 1e-13
        filtered_triplets = [
            (i, j, v) for (i, j), v in triplet_dict.items() if abs(v) > tolerance
        ]

        n_entries = len(filtered_triplets)
        data = np.empty(n_entries, dtype=float)
        row = np.empty(n_entries, dtype=int)
        col = np.empty(n_entries, dtype=int)

        for idx, (i, j, value) in enumerate(filtered_triplets):
            row[idx] = i
            col[idx] = j
            data[idx] = value

        # Single conversion: COO -> CSC
        n_dof = dof_space.total_dofs
        K_csc = sparse.csc_matrix((data, (row, col)), shape=(n_dof, n_dof))

        return K_csc

    # ============================================================
    # DENSE ASSEMBLY
    # ============================================================
    else:
        # Assemble the global stiffness matrix
        # print("\n- Assembling local stiffness matrix into global stiffness matrix")
        for element_index in range(num_elements):
            # Generate the local stiffness matrix for a one-dimensional spring element
            label = mesh.element_property_labels[element_index]
            elem_prop = element_properties[label]

            # Compute local stiffness matrix
            local_K = _compute_local_stiffness(
                elem_prop, element_index, element_connectivity, mesh
            )

            # Get local to global DOF mapping using DOFSpace
            element_nodes = element_connectivity[element_index]
            dof_mapping = dof_space.get_dof_mapping(element_nodes)

            # Assemble the local stiffness matrix into the global stiffness matrix
            for i, global_i in enumerate(dof_mapping):
                for j, global_j in enumerate(dof_mapping):
                    global_stiffness_matrix[global_i, global_j] += local_K[i, j]

        return global_stiffness_matrix


def _compute_local_stiffness(
    elem_prop: ElementProperty,
    element_index: int,
    element_connectivity: list,
    mesh: Mesh,
) -> np.ndarray:
    """
    Compute local stiffness matrix for a given element type.

    Args:
        elem_prop: Element properties
        element_index: Index of current element
        element_connectivity: Element connectivity array
        points: Nodal coordinates

    Returns:
        Local stiffness matrix
    """

    # Generate local stiffness matrix based on element type
    if elem_prop.kind == "spring_1D":
        k_e = param(elem_prop, "k", float)

        # print(f"\n-- Element {element_index}, k = {k_e}")
        local_stiffness_matrix = np.array([[k_e, -k_e], [-k_e, k_e]])
        return local_stiffness_matrix

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
        return local_stiffness_matrix

    elif elem_prop.kind == "bar_2D":
        E = param(elem_prop, "E", float)
        A = param(elem_prop, "A", float)

        # Get element nodes and compute length
        element_nodes = element_connectivity[element_index]
        node1, node2 = element_nodes
        P1 = mesh.points[node1]
        P2 = mesh.points[node2]

        L = np.sqrt((P2[0] - P1[0]) ** 2 + (P2[1] - P1[1]) ** 2)

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
        return local_stiffness_matrix

    else:
        raise ValueError(f"Unknown element kind: {elem_prop.kind}")
