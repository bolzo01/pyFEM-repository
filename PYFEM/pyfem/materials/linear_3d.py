#!/usr/bin/env python
"""
Linear elastic constitutive model for 3D solid mechanics.

Provides:
    stress = D * strain
    D = constitutive stiffness matrix

Strain vector format (engineering shear):
    [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx]

Stress vector format:
    [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_zx]

Created: 2025/11/22 23:41:41
Last modified: 2025/11/23 02:31:13
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from .material import Material


class LinearElastic3D(Material):
    """
    Isotropic 3D linear elastic material.
    """

    def __init__(self, E: float, nu: float):
        self.E = float(E)
        self.nu = float(nu)

    @property
    def constitutive_matrix(self) -> np.ndarray:
        E, nu = self.E, self.nu

        coeff = E / ((1 + nu) * (1 - 2 * nu))
        D = coeff * np.array(
            [
                [1.0 - nu, nu, nu, 0.0, 0.0, 0.0],
                [nu, 1.0 - nu, nu, 0.0, 0.0, 0.0],
                [nu, nu, 1.0 - nu, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5 - nu, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5 - nu, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 - nu],
            ]
        )

        return D

    def stress(self, strain: float | np.ndarray) -> np.ndarray:
        # Reject scalar strain: invalid for 3D material
        if isinstance(strain, (float, int)):
            raise ValueError(
                "LinearElastic3D expects a strain vector "
                "[eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx], "
                f"received {strain}"
            )

        # Validate length = 6
        if not isinstance(strain, np.ndarray) or strain.size != 6:
            raise ValueError(
                "LinearElastic3D expects a strain vector "
                "[eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_zx], "
                f"received {strain}"
            )

        return self.constitutive_matrix @ strain

    def tangent(self, strain: float | np.ndarray) -> np.ndarray:
        return self.constitutive_matrix
