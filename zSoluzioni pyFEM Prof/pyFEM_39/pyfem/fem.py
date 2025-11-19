#!/usr/bin/env python
"""
Module for FEA procedures.

Created: 2025/10/08 17:11:28
Last modified: 2025/11/11 23:03:34
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np
from scipy import sparse

from .dof_types import DOFSpace
from .element_properties import ElementProperties, ElementProperty, param
from .mesh import Mesh
from .quadrature import get_quadrature_rule


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
        mesh: Mesh object with nodal coordinates

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
        # Check if numerical integration is requested
        integration_scheme = elem_prop.meta.get("integration", "analytical")
        if not isinstance(integration_scheme, (str, int)):
            raise TypeError("integration must be a string or integer")

        E = param(elem_prop, "E", float)
        A = param(elem_prop, "A", float)

        # Get element nodes and compute length
        element_nodes = element_connectivity[element_index]
        node1, node2 = element_nodes
        x1 = mesh.points[node1]
        x2 = mesh.points[node2]
        L = x2 - x1

        # Bar stiffness matrix
        if integration_scheme == "analytical":
            # Analytical integration (exact for constant E, A)
            k_e = (E * A) / L
            local_stiffness_matrix = np.array([[k_e, -k_e], [-k_e, k_e]])
            return local_stiffness_matrix
        else:
            # Numerical integration
            x_nodes = mesh.points[element_nodes]
            return _compute_bar_1D_isoparametric(E, A, x_nodes, integration_scheme)

    elif elem_prop.kind == "bar3_1D":
        # Check if numerical integration is requested
        integration_scheme = elem_prop.meta.get("integration", "analytical")
        if not isinstance(integration_scheme, (str, int)):
            raise TypeError("integration must be a string or integer")

        E = param(elem_prop, "E", float)
        A = param(elem_prop, "A", float)

        # Get element nodes and compute length
        element_nodes = element_connectivity[element_index]

        node1, node2, _ = element_nodes
        x1 = mesh.points[node1]
        x2 = mesh.points[node2]
        L = x2 - x1

        # Bar stiffness matrix
        if integration_scheme == "analytical":
            # Closed-form stiffness matrix (valid for constant E, A and a straight element with centered mid-node)
            k_e = (E * A) / L
            local_stiffness_matrix = (
                k_e
                * np.array(
                    [
                        [7.0, 1.0, -8.0],
                        [1.0, 7.0, -8.0],
                        [-8.0, -8.0, 16.0],
                    ]
                )
                / 3.0
            )
            return local_stiffness_matrix
        else:
            # Numerical integration
            x_nodes = mesh.points[element_nodes]
            return _compute_bar3_1D_isoparametric(E, A, x_nodes, integration_scheme)

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


def _compute_bar_1D_isoparametric(
    E: float, A: float, x_nodes: np.ndarray, integration_scheme: str | int
) -> np.ndarray:
    """
    Numerical integration of a 2-node linear 1D bar element stiffness matrix,
    using shape functions and derivatives expressed in the reference space.
    """

    if x_nodes.size != 2:
        raise ValueError("This formulation only applies to 2-node linear elements.")

    # Get Gauss rule
    quad = get_quadrature_rule("bar_1D", integration_scheme)
    if quad is None:
        raise ValueError("Analytical integration should not call numerical routine.")

    K_local = np.zeros((2, 2))

    # Quadrature points in reference space
    xi = quad.points
    w = quad.weights
    n_gauss = quad.n_points

    for ip in range(n_gauss):
        xi_ip = float(xi[ip])
        weight = float(w[ip])

        # Compute B matrix & Jacobian in physical space
        B, J = bar_1D_B_matrix_isoparametric(x_nodes, xi_ip)

        # Integrand at this quadrature point
        integrand = B.T @ (E * A * B)

        # Accumulate contribution
        K_local += weight * integrand * J

    return K_local


def bar_1D_B_matrix_isoparametric(
    x_nodes: np.ndarray, xi: float
) -> tuple[np.ndarray, float]:
    """
    Compute Jacobian and B-matrix for isoparametric 2-node linear bar element.

    Args:
        x_nodes: Array of nodal coordinates [x1, x2]
        xi: Reference coordinate in [-1, 1]

    Returns:
        B: Strain-displacement 1x2 matrix: [dN1/dx, dN2/dx]
        J: Determinant of the Jacobian matrix

    """

    # Derivatives of linear shape functions w.r.t. xi: [dN1/dxi, dN2/dxi]
    dN1_dxi = -0.5
    dN2_dxi = +0.5
    dN_dxi = np.array([[dN1_dxi, dN2_dxi]])  # 1x2 matrix

    # Jacobian: J = dx/dxi = dNi/dxi * xi (= L/2 for 2-node 1D line element only)
    J = float(dN_dxi @ x_nodes.reshape(2, 1))

    if J <= 0:
        raise ValueError(
            f"Jacobian is non-positive (J = {J}). "
            "This indicates reversed node ordering or an inverted element."
        )

    # Transform to physical coordinates: dN/dx = (dN/dxi) * (dxi/dx) = (dN/dxi) / J
    dN_dx = dN_dxi / J

    # B-matrix for 1D 2-node line element: [dN1/dx, dN2/dx]
    B = dN_dx.reshape(1, 2)

    return B, J


def _compute_bar3_1D_isoparametric(
    E: float, A: float, x_nodes: np.ndarray, integration_scheme: str | int
) -> np.ndarray:
    """
    Numerical integration of a 3-node linear 1D bar element stiffness matrix,
    using shape functions and derivatives expressed in the reference space.
    """

    if x_nodes.size != 3:
        raise ValueError("This formulation only applies to 3-node linear elements.")

    # Get Gauss rule
    quad = get_quadrature_rule("bar3_1D", integration_scheme)
    if quad is None:
        raise ValueError("Analytical integration should not call numerical routine.")

    K_local = np.zeros((3, 3))

    # Quadrature points in reference space
    xi = quad.points
    w = quad.weights
    n_gauss = quad.n_points

    for ip in range(n_gauss):
        xi_ip = float(xi[ip])
        weight = float(w[ip])

        # Compute B matrix & Jacobian in physical space
        B, J = bar3_1D_B_matrix_isoparametric(x_nodes, xi_ip)

        # Integrand at this quadrature point
        integrand = B.T @ (E * A * B)

        # Accumulate contribution
        K_local += weight * integrand * J

    return K_local


def bar3_1D_B_matrix_isoparametric(
    x_nodes: np.ndarray, xi: float
) -> tuple[np.ndarray, float]:
    """
    Compute Jacobian and B-matrix for isoparametric 3-node linear bar element.

    Args:
        x_nodes: Array of nodal coordinates [x1, x2, x3]
        xi: Reference coordinate in [-1, 1]

    Returns:
        B: Strain-displacement 1x3 matrix: [dN1/dx, dN2/dx, dN3/dx]
        J: Determinant of the Jacobian matrix

    """

    # Derivatives of linear shape functions w.r.t. xi: [dN1/dxi, dN2/dxi, dN3/dxi]
    dN1_dxi = xi - 0.5
    dN2_dxi = xi + 0.5
    dN3_dxi = -2.0 * xi
    dN_dxi = np.array([[dN1_dxi, dN2_dxi, dN3_dxi]])  # 1x3 matrix

    # Jacobian: J = dx/dxi = dNi/dxi * xi
    J = float(dN_dxi @ x_nodes.reshape(3, 1))

    if J <= 0:
        raise ValueError(
            f"Jacobian is non-positive (J = {J}). "
            "This indicates reversed node ordering or an inverted element."
        )

    # Transform to physical coordinates: dN/dx = (dN/dxi) * (dxi/dx) = (dN/dxi) / J
    dN_dx = dN_dxi / J

    # B-matrix for 1D 3-node line element: [dN1/dx, dN2/dx, dN3/dx]
    B = dN_dx.reshape(1, 3)

    return B, J
