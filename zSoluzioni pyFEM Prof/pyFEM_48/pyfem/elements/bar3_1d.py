#!/usr/bin/env python
"""
Class for bar3_1D element.

Created: 2025/11/17 02:07:24
Last modified: 2025/11/17 23:46:32
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from ..quadrature import get_quadrature_rule
from .element_registry import register_element
from .finite_elements import FiniteElement


@register_element("bar3_1D")
class Bar3_1D(FiniteElement):
    """
    3-node bar element in 1D.

    Supports:
      - Analytical closed‐form stiffness / strain / stress
      - Numerical integration (Gauss) for stiffness / strain / stress
    """

    def __init__(self, params, meta):
        self.A = float(params["A"])
        self.integration = meta.get("integration", "analytical")

    # -----------------------------------------------------------
    # Required element info
    # -----------------------------------------------------------

    @property
    def num_nodes(self):
        return 3

    @property
    def dofs_per_node(self):
        return 1  # 1 DOF per node in 1D

    # -----------------------------------------------------------
    # Shape functions and derivatives
    # -----------------------------------------------------------

    def shape_functions(self, xi):
        # Quadratic Lagrange: nodes at xi = -1, 0, +1
        N1 = 0.5 * xi * (xi - 1.0)
        N2 = 0.5 * xi * (xi + 1.0)
        N3 = 1.0 - xi**2
        return np.array([N1, N2, N3])

    def shape_function_derivatives(self, xi):
        # Derivatives of the above
        dN1 = xi - 0.5
        dN2 = xi + 0.5
        dN3 = -2.0 * xi
        return np.array([dN1, dN2, dN3])

    # -----------------------------------------------------------
    # Integration rule
    # -----------------------------------------------------------

    def integration_points(self):
        if self.integration == "analytical":
            # Analytical stiffness is only valid for perfectly centered mid-node
            return []
        quad = get_quadrature_rule("bar3_1D", self.integration)
        return list(zip(quad.points, quad.weights))

    # -----------------------------------------------------------
    # Jacobian
    # -----------------------------------------------------------

    def jacobian(self, x_nodes, xi):
        dN_dxi = self.shape_function_derivatives(xi)
        J = dN_dxi @ x_nodes

        if J <= 0:
            raise ValueError(
                f"Invalid Jacobian J={J}. Check node ordering or mesh distortion."
            )
        return float(J)

    # -----------------------------------------------------------
    # B matrix
    # -----------------------------------------------------------

    def B_matrix(self, x_nodes, xi):
        dN_dxi = self.shape_function_derivatives(xi)
        J = self.jacobian(x_nodes, xi)
        dN_dx = dN_dxi / J
        return dN_dx.reshape(1, 3)

    # -----------------------------------------------------------
    # Stiffness computation
    # -----------------------------------------------------------

    def compute_stiffness(self, material, x_nodes):
        # Analytical stiffness
        if self.integration == "analytical":
            # Closed-form valid only for the special straight element
            x1, x2, x3 = x_nodes
            L = x2 - x1

            # Check mid-node is centered: x3 = (x1 + x2)/2
            mid = 0.5 * (x1 + x2)
            if abs(x3 - mid) > 1e-12:
                raise ValueError(
                    "Analytical integration for bar3_1D requires the mid-node "
                    "to be exactly centered."
                )
            E = float(material.E)
            k = (E * self.A) / L
            K = (
                k
                * np.array(
                    [
                        [7.0, 1.0, -8.0],
                        [1.0, 7.0, -8.0],
                        [-8.0, -8.0, 16.0],
                    ]
                )
                / 3.0
            )
            return K

        # Numerical integration
        K = np.zeros((3, 3))
        quad = self.integration_points()

        E = float(material.E)
        D = np.array([[E]])  # 1x1 constitutive matrix

        for xi, w in quad:
            B = self.B_matrix(x_nodes, xi)
            J = self.jacobian(x_nodes, xi)
            K += B.T @ D @ B * w * J * self.A

        return K

    # -----------------------------------------------------------
    # Strain & Stress computation
    # -----------------------------------------------------------

    def compute_strain(self, x_nodes, u_nodes):
        """
        Return strains at:
            - the element center if analytical
            - all Gauss points if numerical

        Returns:
            float or np.ndarray  (ngp,)
        """
        if self.integration == "analytical":
            # strain at xi = 0 (element center)
            B = self.B_matrix(x_nodes, xi=0.0)
            eps = float(B @ u_nodes)
            return eps

        # Numerical GP strains
        quad = self.integration_points()
        strains = []

        for xi, _ in quad:
            B = self.B_matrix(x_nodes, xi)
            strains.append(float(B @ u_nodes))

        return np.array(strains)

    def compute_stress(self, material, x_nodes, u_nodes):
        """
        Stress = material.stress(strain)

        Returns:
            float (analytical)
            OR
            np.ndarray (ngp,) for numerical integration
        """
        strain = self.compute_strain(x_nodes, u_nodes)

        # Analytical: scalar
        if np.isscalar(strain):
            return material.stress(strain)

        # Numerical: vector of stresses at GP
        return np.array([material.stress(eps) for eps in strain])
