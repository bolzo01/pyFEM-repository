#!/usr/bin/env python
"""
Module defining the Solution class for FEA results.

Created: 2025/11/13 23:18:15
Last modified: 2025/11/16 01:55:24
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from dataclasses import dataclass, field

import numpy as np

from .dof_types import DOFType


@dataclass
class Solution:
    """Container for finite element analysis solution results.

    Attributes:
        nodal_displacements: Displacement vector for all DOFs
        reaction_forces: Reaction forces at constrained DOFs (computed from K*u - f)
        solver_stats: Optional dictionary of solver performance metrics
    """

    nodal_displacements: np.ndarray
    reaction_forces: np.ndarray | None = None
    dof_types: list[DOFType] = field(default_factory=list)
    solver_stats: dict[str, int | float] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Validate solution data."""
        if not isinstance(self.nodal_displacements, np.ndarray):
            raise TypeError("nodal_displacements must be a numpy array")
        if self.nodal_displacements.ndim != 1:
            raise ValueError("nodal_displacements must be a 1D array")
        if np.any(~np.isfinite(self.nodal_displacements)):
            raise ValueError("nodal_displacements contains NaN or Inf")

        if self.reaction_forces is not None:
            if not isinstance(self.reaction_forces, np.ndarray):
                raise TypeError("reaction_forces must be a numpy array")
            if self.reaction_forces.ndim != 1:
                raise ValueError("reaction_forces must be a 1D array")
            if len(self.reaction_forces) != len(self.nodal_displacements):
                raise ValueError(
                    "reaction_forces must have same length as nodal_displacements"
                )
            if np.any(~np.isfinite(self.reaction_forces)):
                raise ValueError("reaction_forces contains NaN or Inf")

        if self.dof_types and len(self.dof_types) != len(self.nodal_displacements):
            raise ValueError("dof_types list must match number of global DOFs")

    # -------------------------------------------------------------------------
    @property
    def num_dofs(self) -> int:
        """Total number of degrees of freedom in solution."""
        return len(self.nodal_displacements)

    # -------------------------------------------------------------------------
    @property
    def has_reactions(self) -> bool:
        """Return True if reaction forces have been computed."""
        return self.reaction_forces is not None

    # -------------------------------------------------------------------------
    def get_reaction_force(self, dof: int) -> float:
        """Return reaction force at a specific global DOF index."""
        if self.reaction_forces is None:
            raise ValueError("Reaction forces have not been computed")
        if not (0 <= dof < len(self.reaction_forces)):
            raise IndexError(f"DOF {dof} out of range [0, {len(self.reaction_forces)})")
        return float(self.reaction_forces[dof])

    # -------------------------------------------------------------------------
    def compute_equilibrium_residual(
        self,
        K: np.ndarray,
        f: np.ndarray,
    ) -> np.ndarray:
        """
        Compute full global equilibrium residual vector:

            r = K u - f - R
        """
        if K.shape[0] != self.num_dofs:
            raise ValueError("Stiffness matrix size mismatch")
        if f.shape[0] != self.num_dofs:
            raise ValueError("Force vector size mismatch")
        if self.reaction_forces is None:
            raise ValueError("Reaction forces not computed")

        Ku = K @ self.nodal_displacements
        return Ku - f - self.reaction_forces

    # -------------------------------------------------------------------------
    def group_residuals_by_doftype(
        self, residual: np.ndarray
    ) -> dict[DOFType, np.ndarray]:
        """Return residuals grouped by DOF type."""
        if not self.dof_types:
            raise ValueError("No DOF type information available")

        groups: dict[DOFType, list[float]] = {}
        for value, dof_type in zip(residual, self.dof_types):
            groups.setdefault(dof_type, []).append(value)

        return {d: np.array(vals) for d, vals in groups.items()}

    # -------------------------------------------------------------------------
    def check_equilibrium(
        self,
        solver,
        tolerance: float = 1e-10,
        verbose: bool = False,
    ) -> bool:
        """
        Check equilibrium by verifying:

            max(abs(Ku - f - R)) < tolerance

        With verbose=True, prints residual information.
        """
        K = solver.global_stiffness_matrix
        f = solver.global_force_vector
        dof_types = solver.model.dof_space.global_dof_types

        residual = K @ self.nodal_displacements - f - self.reaction_forces

        if verbose:
            print("\n" + "-" * 70)
            print("Equilibrium Verification")
            print("-" * 70)
            print(f"Global residual max = {np.max(np.abs(residual))}")
            self.print_grouped_residuals(residual, dof_types)

        return np.max(np.abs(residual)) < tolerance

    # -------------------------------------------------------------------------
    def print_grouped_residuals(self, residual: np.ndarray, dof_types: list) -> None:
        """Print the residual grouped by DOF type."""
        print("\nResidual by DOF type:")

        # Group values
        groups: dict[DOFType, list[float]] = {}
        for value, dof_type in zip(residual, dof_types):
            groups.setdefault(dof_type, []).append(value)

        # Print max per group
        for dof_type, values in groups.items():
            max_val = float(np.max(np.abs(values)))
            print(f"  {dof_type.value}: max |r| = {max_val:.6e}")

    # -------------------------------------------------------------------------
    def print_reactions(self, dof_space) -> None:
        """Print reaction forces grouped by node and DOF type."""
        if self.reaction_forces is None:
            print("No reaction forces available.")
            return

        print("\n" + "-" * 70)
        print("Reaction Forces")
        print("-" * 70)

        for node in range(len(dof_space.dofs)):
            print(f"Node {node}:")
            for dof_type in dof_space.active_dof_types:
                try:
                    gid = dof_space.get_global_dof(node, dof_type)
                except KeyError:
                    continue
                r = self.get_reaction_force(gid)
                print(f"  {dof_type.value}: {r:.6f}")
