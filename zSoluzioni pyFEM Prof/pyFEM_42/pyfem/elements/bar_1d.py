"""
Class for bar_1D element.

Created: 2025/11/17 01:03:57
Last modified: 2025/11/17 22:46:47
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from ..quadrature import get_quadrature_rule
from .element_registry import register_element
from .finite_elements import FiniteElement


@register_element("bar_1D")
class Bar1D(FiniteElement):
    """
    2-node linear bar element in 1D.

    Supports:
      - Analytical closed-form stiffness
      - Numerical stiffness via Gauss integration
    """

    def __init__(self, params, meta):
        self.E = float(params["E"])
        self.A = float(params["A"])
        self.integration = meta.get("integration", "analytical")

    # -----------------------------------------------------------
    # Required element info
    # -----------------------------------------------------------

    @property
    def num_nodes(self):
        return 2

    @property
    def dofs_per_node(self):
        return 1  # 1 DOF per node in 1D

    # -----------------------------------------------------------
    # Shape functions and derivatives
    # -----------------------------------------------------------

    def shape_functions(self, xi):
        # N1 = (1 - xi)/2 , N2 = (1 + xi)/2
        return np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])

    def shape_function_derivatives(self, xi):
        # dN/dxi = [-1/2, +1/2]
        return np.array([-0.5, 0.5])

    # -----------------------------------------------------------
    # Integration rule
    # -----------------------------------------------------------

    def integration_points(self):
        if self.integration == "analytical":
            # No numerical integration needed
            return []
        else:
            quad = get_quadrature_rule("bar_1D", self.integration)
            return list(zip(quad.points, quad.weights))

    # -----------------------------------------------------------
    # Jacobian
    # -----------------------------------------------------------

    def jacobian(self, x_nodes, xi):
        """
        dx/dxi = dN/dxi * x_nodes
        For 2-node linear bar: J = L/2
        """
        dN_dxi = np.array([-0.5, 0.5])
        J = dN_dxi @ x_nodes
        if J <= 0:
            raise ValueError(f"Invalid Jacobian J={J}. Check node ordering for bar_1D.")
        return float(J)

    # -----------------------------------------------------------
    # B matrix
    # -----------------------------------------------------------

    def B_matrix(self, x_nodes, xi):
        dN_dxi = np.array([-0.5, 0.5])
        J = self.jacobian(x_nodes, xi)
        dN_dx = dN_dxi / J
        return dN_dx.reshape(1, 2)

    # -----------------------------------------------------------
    # Stiffness computation
    # -----------------------------------------------------------

    def compute_stiffness(self, material, x_nodes):
        # Analytical formulation
        if self.integration == "analytical":
            L = x_nodes[1] - x_nodes[0]
            k = (self.E * self.A) / L
            return np.array([[k, -k], [-k, k]])

        # Numerical formulation
        K = np.zeros((2, 2))
        quad = self.integration_points()

        for xi, w in quad:
            B = self.B_matrix(x_nodes, xi)
            J = self.jacobian(x_nodes, xi)
            D = np.array([[self.E]])  # 1x1 constitutive matrix
            K += B.T @ D @ B * w * J * self.A

        return K
