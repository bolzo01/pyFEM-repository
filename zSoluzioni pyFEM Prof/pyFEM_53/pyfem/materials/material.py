#!/usr/bin/env python
"""
Base interface for material (constitutive) models.

This module defines the abstract `Material` class, which all constitutive
models in pyFEM must inherit from. The class specifies the minimal interface
required by finite elements—`stress()`, `tangent()`, and (optionally)
`constitutive_matrix` and `needs_history`.

Concrete material models (e.g., LinearElastic1D) extend this base class.

Created: 2025/10/18 23:11:22
Last modified: 2025/11/17 22:37:18
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from abc import ABC, abstractmethod

import numpy as np


class Material(ABC):
    """
    Abstract base class for all constitutive (material) models.

    A material model provides two fundamental operations:

        stress = stress(strain)
        D = tangent(strain)

    where strain may be a scalar (1D) or a vector/tensor (higher dimensions),
    depending on the element type and material formulation.

    Finite elements will call:
        - material.stress(strain)
        - material.tangent(strain)
        - material.constitutive_matrix  (for small-strain linear cases)

    Subclasses must implement:
        - stress()
        - tangent()

    Notes
    -----
    - For linear materials, ``stress`` and ``tangent`` typically ignore the
      strain input.
    - For nonlinear materials, strain may be scalar or a numpy array.
    - ``needs_history`` can be overridden by materials with internal variables
      (plasticity, damage, viscoelasticity, etc.).
    """

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def stress(self, strain: float | np.ndarray) -> float | np.ndarray:
        """
        Compute the Cauchy stress corresponding to the given strain.

        Parameters
        ----------
        strain : float or np.ndarray
            The current strain measure (scalar for 1D, vector/tensor for nD).

        Returns
        -------
        float or np.ndarray
            The corresponding stress value(s).
        """
        pass

    @abstractmethod
    def tangent(self, strain: float | np.ndarray) -> float | np.ndarray:
        """
        Compute the material tangent stiffness (Jacobian) ∂σ/∂ε.

        Parameters
        ----------
        strain : float or np.ndarray
            The current strain at which the tangent is evaluated.

        Returns
        -------
        float or np.ndarray
            The tangent modulus (scalar for 1D, matrix for nD).
        """
        pass

    # ------------------------------------------------------------------
    # Convenience defaults for simple 1D/linear materials
    # ------------------------------------------------------------------

    @property
    def constitutive_matrix(self) -> np.ndarray:
        """
        Return the constitutive matrix used by small-strain linear elements.

        Default behavior:
            For 1D models, returns a 1x1 matrix D = [ E ] where
            E = tangent(0.0).

        Notes
        -----
        - Higher-dimensional or nonlinear materials should override this
          property if they use a different constitutive matrix structure.
        """
        return np.array([[self.tangent(0.0)]])

    @property
    def needs_history(self) -> bool:
        """
        Whether this material requires internal history variables.

        Returns
        -------
        bool
            True for models such as plasticity or damage that track internal
            state across load steps; False for purely elastic models.

        Notes
        -----
        Override this property and provide the corresponding history-management
        interface in derived classes when implementing nonlinear path-dependent
        materials.
        """
        return False
