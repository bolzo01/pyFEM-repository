#!/usr/bin/env python
"""
Class for 3-node constant-strain triangle element (CST).

Created: 2025/11/17 01:03:57
Last modified: 2025/11/21 23:27:27
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from .element_registry import register_element
from .finite_elements import FiniteElement


@register_element("triangle")
class Triangle(FiniteElement):
    def __init__(self, params, meta):
        # thickness t
        try:
            self.t = float(params["t"])
        except KeyError as exc:
            raise KeyError(
                "Triangle element requires parameter 't' (thickness)."
            ) from exc

    @property
    def num_nodes(self) -> int:
        return 3

    @property
    def dofs_per_node(self) -> int:
        # u_x, u_y per node
        return 2

    def compute_stiffness(self, material, x_nodes) -> np.ndarray:
        """
        Compute the 6x6 element stiffness matrix for a CST element.

        Parameters
        ----------
        material : Material (e.g., LinearElastic2D)
        x_nodes : (3, 2) array of nodal coordinates [[x1,y1],[x2,y2],[x3,y3]]
        """
        if x_nodes.shape != (3, 2):
            raise ValueError(
                f"Triangle expects x_nodes of shape (3, 2), got {x_nodes.shape}"
            )

        t = self.t
        D = material.constitutive_matrix  # 3x3

        B, area = self.B_matrix_triangle(x_nodes)  # B: 3x6
        K_e = B.T @ D @ B * t * area  # 6x6

        return K_e

    # ----------------------------------------------
    # Strain and stress
    # ----------------------------------------------

    def compute_strain(self, x_nodes, u_nodes):
        """
        Compute in-plane strain [eps_xx, eps_yy, gamma_xy] for the CST element.
        """

        if x_nodes.shape != (3, 2):
            raise ValueError(
                f"Triangle expects x_nodes of shape (3, 2), got {x_nodes.shape}"
            )
        if u_nodes.shape not in [(6,), (6, 1)]:
            raise ValueError(
                f"Triangle expects u_nodes of length 6, got shape {u_nodes.shape}"
            )

        B, _ = self.B_matrix_triangle(x_nodes)
        strain = B @ u_nodes.reshape(-1)  # 3-vector

        return strain

    def compute_stress(self, material, x_nodes, u_nodes):
        """
        Compute in-plane stress [sigma_xx, sigma_yy, tau_xy] for the CST element.
        """

        strain = self.compute_strain(x_nodes, u_nodes)
        stress = material.stress(strain)
        return stress

    def B_matrix_triangle(self, x_nodes):
        """B-matrix and area computation for a 3-node CST element."""

        x1, y1 = x_nodes[0]
        x2, y2 = x_nodes[1]
        x3, y3 = x_nodes[2]

        x13, x21, x32 = x1 - x3, x2 - x1, x3 - x2
        y12, y23, y31 = y1 - y2, y2 - y3, y3 - y1

        area = 0.5 * ((x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3) + (x1 * y2 - x2 * y1))
        if area <= 0:
            msg = f"non-positive area for triangle element {self.element_index}: {area}"
            raise ValueError(msg)

        B = (
            0.5
            / area
            * np.array(
                [
                    [y23, 0.0, y31, 0.0, y12, 0.0],
                    [0.0, x32, 0.0, x13, 0.0, x21],
                    [x32, y23, x13, y31, x21, y12],
                ]
            )
        )

        return B, area

    # Mandatory abstract methods (triangle does not use these)

    def shape_functions(self, xi):
        raise NotImplementedError("triangle has no shape functions.")

    def shape_function_derivatives(self, xi):
        raise NotImplementedError("triangle has no shape functions.")

    def integration_points(self):
        # No integration needed for closed-form stiffness
        return []

    def jacobian(self, x_nodes, xi):
        raise NotImplementedError("triangle has no Jacobian.")
