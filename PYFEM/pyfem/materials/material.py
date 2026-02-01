#!/usr/bin/env python
"""
Base interface for material (constitutive) models.

This module defines the base `Material` class, which all constitutive
models in pyFEM must inherit from, and the more specific
`MechanicalMaterial` class, which is used for structural/mechanical
constitutive laws.

- `Material` is a generic base for *all* materials (mechanical, thermal,
  diffusion, etc.). It does **not** enforce a mechanical interface.

- `MechanicalMaterial` extends `Material` and specifies the minimal
  interface required by mechanical finite elements—`stress()` and
  `tangent()`—and (optionally) `constitutive_matrix` and `needs_history`.

Concrete mechanical material models (e.g., LinearElastic1D) should
inherit from `MechanicalMaterial`.

Non-mechanical models (e.g., Diffusion1D for heat transfer) should
inherit from the generic `Material` base and expose whatever fields
they need (e.g., thermal diffusivity) without being forced to
implement mechanical notions such as stress or tangent.

Created: 2025/10/18 23:11:22
Last modified: 2026/01/29 18:09:31
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from abc import ABC, abstractmethod

import numpy as np


class Material(ABC):
    """
    Abstract base class for all material models (mechanical, thermal, etc.).

    This base class intentionally does **not** enforce any specific
    constitutive interface. It provides a common root type so that
    all materials can be stored in a single registry and passed
    through the same Model API.

    Specific families of materials (e.g., mechanical constitutive laws)
    should derive from more specialized subclasses such as
    `MechanicalMaterial`.
    """

    def __init__(self, name: str | None = None):
        """
        Parameters
        ----------
        name : str, optional
            Optional human-readable name for the material.
        """
        self.name = name

    def describe(self) -> str:
        """
        Return a short text description for debugging/introspection.
        """
        classname = self.__class__.__name__
        return f"{classname}(name={self.name!r})"

    @property
    def needs_history(self) -> bool:
        """
        Whether this material requires internal history variables.

        Returns
        -------
        bool
            True for models such as plasticity or damage that track internal
            state across load steps; False for purely elastic models or
            materials without internal variables.

        Notes
        -----
        Mechanical materials that are path-dependent (plasticity, damage,
        viscoelasticity, etc.) should override this property and provide a
        corresponding history-management interface.
        """
        return False


class MechanicalMaterial(Material, ABC):
    """
    Abstract base class for mechanical constitutive (material) models.

    A mechanical material model provides two fundamental operations:

        stress = stress(strain)
        D = tangent(strain)

    where strain may be a scalar (1D) or a vector/tensor (higher dimensions),
    depending on the element type and material formulation.

    Finite elements will typically call:
        - material.stress(strain)
        - material.tangent(strain)
        - material.constitutive_matrix  (for small-strain linear cases)

    Subclasses must implement:
        - stress()
        - tangent()
    """

    # ------------------------------------------------------------------
    # Required mechanical interface
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
        ...

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
        ...

    # ------------------------------------------------------------------
    # Convenience defaults for simple 1D/linear mechanical materials
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
        - For nonlinear materials, evaluating tangent(0.0) is often not
          meaningful; such materials should override this property to
          provide an appropriate representation or raise an error.
        """
        return np.array([[self.tangent(0.0)]])

    @property
    def needs_history(self) -> bool:
        """
        Whether this mechanical material requires internal history variables.

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


class Diffusion1D(Material):
    """
    Simple 1D thermal diffusion material model.

    This material is used in heat-transfer problems governed by

        ∂T/∂t = α ∂²T/∂x² + s(x)

    Parameters
    ----------
    alpha : float
        Thermal diffusivity coefficient [m²/s].

    Notes
    -----
    - This class is used by diffusion-type solvers, which read `alpha`
      directly when assembling the diffusion operator.
    - Source terms s(x) are handled at the element/solver level.
    - Diffusion1D does *not* define mechanical notions such as stress or
      tangent; it is purely a thermal material and therefore inherits from
      the generic `Material` base, not `MechanicalMaterial`.
    """

    def __init__(self, alpha: float):
        super().__init__(name="Diffusion1D")
        self.alpha = float(alpha)

    def __repr__(self) -> str:
        return f"Diffusion1D(alpha={self.alpha})"


class Diffusion2D(Material):
    """
    Simple 2D thermal diffusion material model.

    This material is used in heat-transfer problems governed by

        ρ c (∂T/∂t) = ∇(D∇T) + s

    Parameters
    ----------
    ρ : float
        Density [kg/m³]
    c : float
        specific heat capacity [J/(kg*K)]
    s : float
        heat source per unit volume [K/s or °C/s]
    D : np.array
        conductivity matrix - for isotropic materials, D = k * I where k is condutivity [W/(m*K)]
    alpha : float
        Thermal diffusivity coefficient [m²/s].

    Notes
    -----
    - This class is used by diffusion-type solvers, which read `alpha`
      directly when assembling the diffusion operator.
    - Source terms s(x) are handled at the element/solver level.
    - Diffusion2D does *not* define mechanical notions such as stress or
      tangent; it is purely a thermal material and therefore inherits from
      the generic `Material` base, not `MechanicalMaterial`.
    """

    def __init__(self, rho: float, c: float, k: float, alpha: float):
        super().__init__(name="Diffusion2D")
        self.rho = float(rho)
        self.c = float(c)
        self.k = float(k)
        self.alpha = float(alpha)

    def __repr__(self) -> str:
        return (
            f"Diffusion2D(rho={self.rho}, c={self.c}, k={self.k}, alpha={self.alpha})"
        )
