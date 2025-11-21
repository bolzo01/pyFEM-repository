#!/usr/bin/env python
"""
Class for spring_1D element.

Created: 2025/11/16 19:18:04
Last modified: 2025/11/17 23:58:21
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from .element_registry import register_element
from .finite_elements import FiniteElement


@register_element("spring_1D")
class Spring1D(FiniteElement):
    def __init__(self, params, meta):
        self.k = float(params["k"])

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def dofs_per_node(self) -> int:
        return 1

    def compute_stiffness(self, material=None, x_nodes=None) -> np.ndarray:
        k = self.k
        return np.array([[k, -k], [-k, k]])

    def compute_stress(self, material, x_nodes, u_nodes):
        return None

    # Mandatory abstract methods (spring_1D does not use these)

    def shape_functions(self, xi):
        raise NotImplementedError("spring_1D has no shape functions.")

    def shape_function_derivatives(self, xi):
        raise NotImplementedError("spring_1D has no shape functions.")

    def integration_points(self):
        # No integration needed for closed-form stiffness
        return []

    def jacobian(self, x_nodes, xi):
        raise NotImplementedError("spring_1D has no geometry or Jacobian.")
