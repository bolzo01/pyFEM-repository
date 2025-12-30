#!/usr/bin/env python
"""
1D heat diffusion element (2-node linear bar).

Implements the 1D transient heat equation:
    ∂T/∂t = α ∂²T/∂x² + s(x,t)

where:
    T = temperature [K or °C]
    α = thermal diffusivity [m²/s]
    s = heat source per unit volume [K/s or °C/s]

Created: 2025/12/08 09:01:27
Last modified: 2025/12/17 00:28:03
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from .element_registry import register_element
from .finite_elements import FiniteElement


@register_element("bar_1D_heat")
class Bar1DHeat(FiniteElement):
    """
    2-node linear element for 1D transient heat diffusion.

    Element matrices:
        Mass matrix:       M_e = (ρcL/6) * [[2, 1], [1, 2]]
        Stiffness matrix:  K_e = (kA/L) * [[1, -1], [-1, 1]]
        Source vector:     f_e = (sL/2) * [1, 1]^T

    where:
        ρ = density [kg/m³]
        c = specific heat capacity [J/(kg·K)]
        k = thermal conductivity [W/(m·K)]
        A = cross-sectional area [m²]
        L = element length [m]
        s = volumetric heat source [W/m³] = [J/(m³·s)]

    diffusivity: α = k/(ρc)
    """

    def __init__(self, params, meta):
        """
        Initialize heat element.

        Required params:
            'A': cross-sectional area [m²] (optional, default=1.0)

        Optional params:
            'source': volumetric heat source [K/s] (default=0.0)
                      Note: This is s*(A/ρc) in dimensional analysis
        """
        self.A = float(params.get("A", 1.0))
        self.source = float(params.get("source", 0.0))

    # -----------------------------------------------------------
    # Required element interface
    # -----------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def dofs_per_node(self) -> int:
        return 1  # Temperature at each node

    # -----------------------------------------------------------
    # Diffusion-specific methods
    # -----------------------------------------------------------

    def compute_diffusion_matrices(
        self, x_nodes: np.ndarray, material, params: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute element mass and stiffness matrices for diffusion.

        Args:
            x_nodes: Nodal coordinates [x1, x2]
            material: Diffusion1D material with 'alpha' attribute
            params: Element parameters

        Returns:
            (M_e, K_e): Element mass and stiffness matrices
        """
        # Element length
        x1, x2 = float(x_nodes[0]), float(x_nodes[1])
        L = x2 - x1

        if L <= 0:
            raise ValueError(f"Element {self.element_index}: invalid length L = {L}")

        # Material property: thermal diffusivity
        if material is None:
            raise ValueError(
                f"Element {self.element_index}: material required for heat diffusion"
            )

        alpha = getattr(material, "alpha", None)
        if alpha is None:
            raise ValueError(
                f"Element {self.element_index}: material must have 'alpha' "
                f"(thermal diffusivity) attribute"
            )

        # Mass matrix: M_e = (L/6) * [[2,1],[1,2]]
        # Note: This includes ρc implicitly through the formulation
        M_e = (L / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])

        # Stiffness matrix: K_e = (α/L) * [[1,-1],[-1,1]]
        K_e = (alpha / L) * np.array([[1.0, -1.0], [-1.0, 1.0]])

        return M_e, K_e

    def compute_source_vector(
        self, x_nodes: np.ndarray, params: dict, time: float = 0.0
    ) -> np.ndarray:
        """
        Compute element source vector from heat generation.

        Args:
            x_nodes: Nodal coordinates [x1, x2]
            params: Element parameters (contains 'source')
            time: Current time (for time-dependent sources)

        Returns:
            f_e: Element source vector [2]
        """
        x1, x2 = float(x_nodes[0]), float(x_nodes[1])
        L = x2 - x1

        # Get source term (could be callable for time-dependence)
        source = params.get("source", 0.0)
        if callable(source):
            s = source(time)
        else:
            s = float(source)

        # Source vector: f_e = (sL/2) * [1, 1]^T
        f_e = (s * L / 2.0) * np.array([1.0, 1.0])

        return f_e

    # -----------------------------------------------------------
    # Mechanics-related methods (not used for heat transfer
    # ...mandatory FiniteElement abstract methods)
    # -----------------------------------------------------------

    def compute_stiffness(self, material=None, x_nodes=None) -> np.ndarray:
        """Not used for heat transfer problems."""
        raise NotImplementedError(
            "bar_1D_heat is for thermal analysis only. "
            "Use compute_diffusion_matrices() instead."
        )

    def compute_stress(self, material, x_nodes, u_nodes):
        """Not applicable for heat transfer."""
        raise NotImplementedError(
            "Stress computation not applicable for heat transfer elements"
        )

    def shape_functions(self, xi):
        """Not used for heat transfer problems."""
        raise NotImplementedError("Not applicable for heat transfer elements")

    def shape_function_derivatives(self, xi):
        """Not used for heat transfer problems."""
        raise NotImplementedError("Not applicable for heat transfer elements")

    def integration_points(self):
        # Not needed because compute_diffusion_matrices is closed-form
        raise NotImplementedError("Not applicable for heat transfer elements")

    def jacobian(self, x_nodes, xi):
        """Not used for heat transfer problems."""
        raise NotImplementedError("Not applicable for heat transfer elements")
