#!/usr/bin/env python
"""
2D heat diffusion element (heat exchanger with triangular element).

Implements the 2D transient heat equation:
    ρ c (∂T/∂t) = ∇(D∇T) + s

where:
    T = temperature [K or °C]
    ρ = density [kg/m³]
    c = specific heat capacity [J/(kg*K)]
    s = heat source per unit volume [K/s or °C/s]

Created: 2025/12/08 09:01:27
Last modified: 2026/02/04 15:29:30
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from .element_registry import register_element
from .finite_elements import FiniteElement


@register_element("triangle_heat")
class Triangle_heat(FiniteElement):
    """
    3-node linear triangular element for 2D transient heat diffusion.

    Element matrices:
        Mass matrix:       M_e = (ρ * c * Area / 12) * [[2, 1, 1],
                                                        [1, 2, 1],
                                                        [1, 1, 2]]

        Stiffness matrix:  K_e = (k * t * Area) * (B.T @ B)
                        (B has shape functions derivatives)

        Source vector:     f_e = (s * Area / 3) * [1, 1, 1]^T

    where:
        ρ    = density [kg/m^3]
        c    = specific heat capacity [J/(kg*K)]
        k    = thermal conductivity [W/(m*K)]
        Area = area of the triangular element [m^2]
        t    = thickness of the element (often 1.0) [m]
        s    = volumetric heat source [W/m^3]
        B    = gradient matrix (depends on node coordinates)

    diffusivity: alpha = k/(ρ * c) [m^2/s]
    """

    def __init__(self, params, meta):
        """
        Initialize heat element.

        Required params:
            't': thickness [m] (optional, default=1.0)

        Optional params:
            'source': volumetric heat source [K/s] (default=0.0)
        """
        # thickness t
        try:
            self.t = float(params["t"])
        except KeyError as exc:
            raise KeyError(
                "Triangle element requires parameter 't' (thickness)."
            ) from exc

        self.source = float(params.get("source", 0.0))

    # -----------------------------------------------------------
    # Required element interface
    # -----------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return 3

    @property
    def dofs_per_node(self) -> int:
        return 1  # Temperature at each node

    # -----------------------------------------------------------
    # Diffusion-specific methods
    # -----------------------------------------------------------

    def compute_diffusion_matrices(
        self, coords: np.ndarray, material, params: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute element mass and stiffness matrices for diffusion.

        Args:
            coords: Nodal coordinates [(x1, y1), (x2, y2), (x3, y3)]
            material: Diffusion2D material with 'alpha' attribute
            params: Element parameters

        Returns:
            (M_e, K_e): Element mass and stiffness matrices
        """

        B, A = self.B_matrix_triangle_heat(coords)  # B: 2x3

        if A <= 0:
            raise ValueError(f"Element {self.element_index}: invalid area A = {A}")

        if material is None:
            raise ValueError(
                f"Element {self.element_index}: material required for heat diffusion"
            )

        # Material property: thermal diffusivity
        alpha = getattr(material, "alpha", None)
        if alpha is None:
            raise ValueError(
                f"Element {self.element_index}: material must have 'alpha' "
                f"(thermal diffusivity [m^2/s]) attribute"
            )

        t = params.get("t", 1.0)

        if t is None:
            raise ValueError(
                f"Element {self.element_index}: material must have 't' "
                f"(thermal diffusivity [m^2/s]) attribute"
            )

        # Material property: density
        rho = getattr(material, "rho", None)
        if rho is None:
            raise ValueError(
                f"Element {self.element_index}: material must have 'rho' "
                f"(density [kg/m^3]) attribute"
            )

        # Material property: specific heat capacity
        c = getattr(material, "c", None)
        if c is None:
            raise ValueError(
                f"Element {self.element_index}: material must have 'c' "
                f"(specific heat capacity [J/(kg*K)]) attribute"
            )

        # Material property: thermal conductivity
        k = getattr(material, "c", None)
        if k is None:
            raise ValueError(
                f"Element {self.element_index}: material must have 'k' "
                f"(thermal conductivity [W/(m*K)]) attribute"
            )

        # Mass matrix: M_e = (ρ * c * Area / 12) * [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
        # Note: This includes ρc implicitly through the formulation
        M_e = (rho * c * A / 12.0) * np.array(
            [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]
        )

        # Stiffness matrix: K_e = (k * t * Area) * (B.T @ B)
        K_e = (k * t * A) * (B.T @ B)

        return M_e, K_e

    def compute_source_vector(
        self, coords: np.ndarray, params: dict, time: float = 0.0
    ) -> np.ndarray:
        """
        Compute element source vector from heat generation.

        Args:
            coords: Nodal coordinates [(x1, y1), (x2, y2), (x3, y3)]
            params: Element parameters (contains 'source')
            time: Current time (for time-dependent sources)

        Returns:
            f_e: Element source vector [2]
        """
        B, A = self.B_matrix_triangle_heat(coords)  # B: 2x3

        if A <= 0:
            raise ValueError(f"Element {self.element_index}: invalid area A = {A}")

        # Get source term (could be callable for time-dependence)
        source = params.get("source", 0.0)
        if callable(source):
            s = source(time)
        else:
            s = float(source)

        # Source vector: f_e = (s * Area / 3) * [1, 1, 1].T
        f_e = (s * A * self.t / 3.0) * np.array([1.0, 1.0, 1.0])

        return f_e

    def B_matrix_triangle_heat(self, coords: np.ndarray):
        """B-matrix and area computation for a 3-node CST element."""

        x1, y1 = coords[0, 0], coords[0, 1]
        x2, y2 = coords[1, 0], coords[1, 1]
        x3, y3 = coords[2, 0], coords[2, 1]

        x13, x21, x32 = x1 - x3, x2 - x1, x3 - x2
        y12, y23, y31 = y1 - y2, y2 - y3, y3 - y1

        area = 0.5 * ((x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3) + (x1 * y2 - x2 * y1))
        if area <= 0:
            msg = f"non-positive area for triangle element {self.element_index}: {area}"
            raise ValueError(msg)

        B = (1.0 / (2.0 * area)) * np.array(
            [
                [y23, y31, y12],  # dN/dx
                [x32, x13, x21],  # dN/dy
            ]
        )

        return B, area

    # -----------------------------------------------------------
    # Mechanics-related methods (not used for heat transfer
    # ...mandatory FiniteElement abstract methods)
    # -----------------------------------------------------------

    def compute_stiffness(self, material=None, x_nodes=None) -> np.ndarray:
        """Not used for heat transfer problems."""
        raise NotImplementedError(
            "triangle_heat is for thermal analysis only. "
            "Use compute_diffusion_matrices() instead."
        )

    def compute_stress(self, material, x_nodes, u_nodes):
        """Not applicable for heat transfer."""
        raise NotImplementedError(
            "Stress computation not applicable for heat transfer elements"
        )

    def integration_points(self):
        # Not needed because compute_diffusion_matrices is closed-form
        raise NotImplementedError("Not applicable for heat transfer elements")

    # Mandatory abstract methods (triangle does not use these)

    def shape_functions(self, xi):
        raise NotImplementedError("triangle has no shape functions.")

    def shape_function_derivatives(self, xi):
        raise NotImplementedError("triangle has no shape functions.")

    def jacobian(self, x_nodes, xi):
        raise NotImplementedError("triangle has no Jacobian.")
