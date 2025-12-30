#!/usr/bin/env python
"""
Linear elastic material model (1D).

This module defines a uniaxial linear elastic constitutive model for use with
1D finite elements such as bar_1D. The model provides:

    - stress-strain relation:    stress = E strain
    - tangent stiffness:         d stress / d strain = E

It implements the generic Material interface expected by the finite element
formulations.

Created: 2025/11/16 21:18:34
Last modified: 2025/11/17 22:23:57
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from .material import Material


class LinearElastic1D(Material):
    """
    Uniaxial (1D) linear elastic material.

    Parameters:
        E (float):
            Young's modulus.

    Notes:
        The material model accepts either a scalar strain value or a NumPy array
        of strain values. This allows vectorized evaluation, e.g. computing the
        stress at multiple Gauss points in a single call.
    """

    def __init__(self, E: float):
        self.E = float(E)

    def stress(self, strain: float | np.ndarray) -> float | np.ndarray:
        return self.E * strain

    def tangent(self, strain: float | np.ndarray) -> float | np.ndarray:
        return self.E
