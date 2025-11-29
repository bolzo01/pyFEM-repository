#!/usr/bin/env python
"""
Class for 4-node linear constant-strain tetrahedral element (CST).

Created: 2025/11/23 00:15:34
Last modified: 2025/11/29 12:28:29
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from .element_registry import register_element
from .finite_elements import FiniteElement


@register_element("tetra")
class Tetra(FiniteElement):
    """Linear 4-node constant-strain tetrahedron element."""

    def __init__(self, params, meta):
        pass  # no geometric parameters required

    @property
    def num_nodes(self) -> int:
        return 4

    @property
    def dofs_per_node(self) -> int:
        return 3  # u_x, u_y, u_z

    def compute_stiffness(self, material, x_nodes) -> np.ndarray:
        """
        Compute the 12x12 element stiffness matrix for a tetrahedral element.

        Parameters
        ----------
        material : Material (e.g., LinearElastic3D)
        x_nodes : (4, 3) array of nodal coordinates [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4]]
        """
        if x_nodes.shape != (4, 3):
            raise ValueError(
                f"Tetra expects x_nodes of shape (4, 3), got {x_nodes.shape}"
            )
        if material is None:
            raise ValueError(
                f"Tetra element {self.element_index}: material was not provided."
            )

        D = material.constitutive_matrix  # 6x6

        B, volume = self.B_matrix_tetra(x_nodes)  # B: 4x12
        K_e = B.T @ D @ B * volume  # 12x12

        return K_e

    # ----------------------------------------------
    # Strain and stress
    # ----------------------------------------------

    def compute_strain(self, x_nodes, u_nodes):
        """
        Return the 3D strain vector:
        [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx]
        """
        B, _ = self.B_matrix_tetra(x_nodes)
        strain = B @ u_nodes.reshape(-1)

        return strain

    def compute_stress(self, material, x_nodes, u_nodes):
        strain = self.compute_strain(x_nodes, u_nodes)
        stress = material.stress(strain)
        return stress

    def B_matrix_tetra(self, x_nodes: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute B matrix and volume for a 4-node linear tetrahedral element."""
        # formulas from Carlos Felippa's AFEM.Ch09.pdf
        x1, y1, z1 = x_nodes[0]
        x2, y2, z2 = x_nodes[1]
        x3, y3, z3 = x_nodes[2]
        x4, y4, z4 = x_nodes[3]

        x12 = x1 - x2
        x13 = x1 - x3
        x14 = x1 - x4
        # x23 = x2 - x3
        x24 = x2 - x4
        x34 = x3 - x4
        x21 = x2 - x1
        x31 = x3 - x1
        # x41 = x4 - x1
        x32 = x3 - x2
        x42 = x4 - x2
        x43 = x4 - x3

        y12 = y1 - y2
        y13 = y1 - y3
        y14 = y1 - y4
        y23 = y2 - y3
        y24 = y2 - y4
        y34 = y3 - y4
        y21 = y2 - y1
        y31 = y3 - y1
        # y41 = y4 - y1
        y32 = y3 - y2
        y42 = y4 - y2
        y43 = y4 - y3

        z12 = z1 - z2
        z13 = z1 - z3
        z14 = z1 - z4
        z23 = z2 - z3
        z24 = z2 - z4
        z34 = z3 - z4
        z21 = z2 - z1
        z31 = z3 - z1
        # z41 = z4 - z1
        z32 = z3 - z2
        z42 = z4 - z2
        z43 = z4 - z3

        a1 = y42 * z32 - y32 * z42
        b1 = x32 * z42 - x42 * z32
        c1 = x42 * y32 - x32 * y42
        a2 = y31 * z43 - y34 * z13
        b2 = x43 * z31 - x13 * z34
        c2 = x31 * y43 - x34 * y13
        a3 = y24 * z14 - y14 * z24
        b3 = x14 * z24 - x24 * z14
        c3 = x24 * y14 - x14 * y24
        a4 = y13 * z21 - y12 * z31
        b4 = x21 * z13 - x31 * z12
        c4 = x13 * y21 - x12 * y31

        determinant_jacobian = (
            +x21 * (y23 * z34 - y34 * z23)
            + x32 * (y34 * z12 - y12 * z34)
            + x43 * (y12 * z23 - y23 * z12)
        )
        volume = determinant_jacobian / 6.0
        if volume <= 0:
            msg = f"non-positive volume for tetrahedral element {self.element_index}: {volume}"
            raise ValueError(msg)

        B = np.array(
            [
                [a1, 0.0, 0.0, a2, 0.0, 0.0, a3, 0.0, 0.0, a4, 0.0, 0.0],
                [0.0, b1, 0.0, 0.0, b2, 0.0, 0.0, b3, 0.0, 0.0, b4, 0.0],
                [0.0, 0.0, c1, 0.0, 0.0, c2, 0.0, 0.0, c3, 0.0, 0.0, c4],
                [b1, a1, 0.0, b2, a2, 0.0, b3, a3, 0.0, b4, a4, 0.0],
                [0.0, c1, b1, 0.0, c2, b2, 0.0, c3, b3, 0.0, c4, b4],
                [c1, 0.0, a1, c2, 0.0, a2, c3, 0.0, a3, c4, 0.0, a4],
            ]
        ) / (6.0 * volume)
        return B, volume

    # Mandatory abstract methods (tetra does not use these)

    def shape_functions(self, xi):
        raise NotImplementedError("tetra has no shape functions.")

    def shape_function_derivatives(self, xi):
        raise NotImplementedError("tetra has no shape functions.")

    def integration_points(self):
        # No integration needed for closed-form stiffness
        return []

    def jacobian(self, x_nodes, xi):
        raise NotImplementedError("tetra has no Jacobian.")
