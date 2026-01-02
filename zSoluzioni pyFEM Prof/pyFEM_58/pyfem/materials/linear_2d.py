#!/usr/bin/env python
"""
Linear elastic constitutive model for 2D solid mechanics in plane stress and plane strain.

Provides:
    stress = D * strain
    D = constitutive stiffness matrix

Supported formulations:
    - plane stress
    - plane strain

Strain vector format:
    [eps_xx, eps_yy, gamma_xy]

Stress vector format:
    [sigma_xx, sigma_yy, tau_xy]

Created: 2025/11/16 21:18:34
Last modified: 2025/12/09 01:53:56
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from .material import MechanicalMaterial


class LinearElastic2D(MechanicalMaterial):
    """
    Isotropic 2D linear elastic material supporting plane stress and plane strain.
    """

    def __init__(self, E: float, nu: float, formulation: str):
        super().__init__(name="LinearElastic1D")
        self.E = float(E)
        self.nu = float(nu)
        if formulation not in {"plane_stress", "plane_strain"}:
            raise ValueError("formulation must be 'plane_stress' or 'plane_strain'")
        self.formulation = formulation

    @property
    def constitutive_matrix(self) -> np.ndarray:
        E, nu = self.E, self.nu

        if self.formulation == "plane_stress":
            coeff = E / (1 - nu**2)
            D = coeff * np.array(
                [
                    [1.0, nu, 0.0],
                    [nu, 1.0, 0.0],
                    [0.0, 0.0, (1.0 - nu) / 2.0],
                ]
            )
            return D

        else:
            # plane strain
            coeff = E / ((1 + nu) * (1 - 2 * nu))
            D = coeff * np.array(
                [
                    [1.0 - nu, nu, 0.0],
                    [nu, 1.0 - nu, 0.0],
                    [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
                ]
            )
            return D

    def stress(self, strain: float | np.ndarray) -> np.ndarray:
        # Reject scalar strain: invalid for 2D material
        if isinstance(strain, (float, int)):
            raise ValueError(
                "LinearElastic2D expects a strain vector [eps_xx, eps_yy, gamma_xy], "
                f"received scalar {strain}"
            )

        # Validate length = 3 (plane stress/strain vector format)
        if not isinstance(strain, np.ndarray) or strain.size != 3:
            raise ValueError(
                "LinearElastic2D expects a strain vector [eps_xx, eps_yy, gamma_xy], "
                f"received {strain}"
            )

        return self.constitutive_matrix @ strain

    def tangent(self, strain: float | np.ndarray) -> np.ndarray:
        return self.constitutive_matrix
