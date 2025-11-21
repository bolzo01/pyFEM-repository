#!/usr/bin/env python
"""
Class for bar_2D element.

Created: 2025/11/17 01:03:57
Last modified: 2025/11/18 00:02:21
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from .element_registry import register_element
from .finite_elements import FiniteElement


@register_element("bar_2D")
class Bar2D(FiniteElement):
    def __init__(self, params, meta):
        self.A = float(params["A"])

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def dofs_per_node(self) -> int:
        return 2

    def compute_stiffness(self, material, x_nodes) -> np.ndarray:
        E = float(material.E)
        A = self.A

        # Get element nodes and compute length
        P1, P2 = x_nodes
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

    # ----------------------------------------------
    # Strain and stress
    # ----------------------------------------------

    def compute_strain(self, x_nodes, u_nodes):
        """
        Compute axial strain:

            eps = (u2_local - u1_local) / L
                 = ( (u2 - u1) ⋅ direction ) / L
        """
        P1, P2 = x_nodes
        dx = P2[0] - P1[0]
        dy = P2[1] - P1[1]
        L = np.sqrt(dx * dx + dy * dy)

        # direction cosines
        c = dx / L
        s = dy / L

        # extract nodal displacements in global coords
        u1 = u_nodes[0:2]
        u2 = u_nodes[2:4]

        # local axial displacement: projection onto bar axis
        u1_local = c * u1[0] + s * u1[1]
        u2_local = c * u2[0] + s * u2[1]

        return (u2_local - u1_local) / L

    def compute_stress(self, material, x_nodes, u_nodes):
        """
        Axial stress = E * eps
        """
        eps = self.compute_strain(x_nodes, u_nodes)
        return material.stress(eps)

    # Mandatory abstract methods (bar_2D does not use these)

    def shape_functions(self, xi):
        raise NotImplementedError("bar_2D has no shape functions.")

    def shape_function_derivatives(self, xi):
        raise NotImplementedError("bar_2D has no shape functions.")

    def integration_points(self):
        # No integration needed for closed-form stiffness
        return []

    def jacobian(self, x_nodes, xi):
        raise NotImplementedError("bar_2D has no geometry or Jacobian.")
