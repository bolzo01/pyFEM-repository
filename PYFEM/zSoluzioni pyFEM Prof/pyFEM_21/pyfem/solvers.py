#!/usr/bin/env python
"""
Module defining the FEA solvers.

Created: 2025/10/18 10:24:33
Last modified: 2025/10/27 09:59:15
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from enum import Enum, auto

import numpy as np

from .dof_types import DOFSpace
from .element_properties import ElementProperties
from .fem import (
    apply_nodal_forces,
    apply_prescribed_displacements,
    assemble_global_stiffness_matrix,
)
from .mesh import Mesh


class SolverState(Enum):
    INITIALIZED = auto()
    ASSEMBLED = auto()
    BOUNDARY_APPLIED = auto()
    SOLVED = auto()


class LinearStaticSolver:
    """Solves linear static finite element problems.

    Assembles and solves the system of equations KU=F.
    """

    def __init__(
        self,
        mesh: Mesh,
        element_properties: ElementProperties,
        applied_forces: list[tuple[int, float]] | None,
        prescribed_displacements: list[tuple[int, float]],
        dof_space: DOFSpace,
    ):
        self.mesh = mesh
        self.element_properties = element_properties
        self.applied_forces = applied_forces
        self.prescribed_displacements = prescribed_displacements
        self.dof_space = dof_space
        self.state = SolverState.INITIALIZED

        # Initialize global matrices and vectors as instance attributes
        total_dofs = self.dof_space.total_dofs
        self.global_stiffness_matrix = np.zeros((total_dofs, total_dofs))
        self.original_global_stiffness_matrix = np.zeros((total_dofs, total_dofs))
        self.global_force_vector = np.zeros(total_dofs)

        # Will be computed by solve()
        self.nodal_displacements: np.ndarray

    def _ensure_state(self, expected: SolverState) -> None:
        if self.state != expected:
            raise RuntimeError(
                f"Invalid solver state: expected {expected.name}, got {self.state.name}"
            )

    def assemble_global_matrix(self) -> None:
        """Constructs the system of equations KU=F."""

        self._ensure_state(SolverState.INITIALIZED)

        # Assemble the global stiffness matrix
        assemble_global_stiffness_matrix(
            self.mesh, self.element_properties, self.global_stiffness_matrix
        )
        print("\n- Global stiffness matrix K:")
        for row in self.global_stiffness_matrix:
            print(row)

        # Save a copy of the original global stiffness matrix before applying boundary conditions
        self.original_global_stiffness_matrix = self.global_stiffness_matrix.copy()

        self.state = SolverState.ASSEMBLED
        return None

    def apply_boundary_conditions(self) -> None:
        """Applies Neumann and Dirichlet boundary conditions."""

        self._ensure_state(SolverState.ASSEMBLED)

        # Boundary conditions: Apply forces
        apply_nodal_forces(self.applied_forces, self.global_force_vector)

        # Boundary conditions: Constrain displacements
        apply_prescribed_displacements(
            self.prescribed_displacements,
            self.global_stiffness_matrix,
            self.global_force_vector,
        )

        print(
            "\n- Modified global stiffness matrix K after applying boundary conditions:"
        )
        for row in self.global_stiffness_matrix:
            print(row)

        print("\n- Global force vector F after applying boundary conditions:")
        print(self.global_force_vector)

        self.state = SolverState.BOUNDARY_APPLIED

        return None

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """Solves the linear system of equations KU=F.

        Returns:
            Solution array and global stiffness matrix before applying boundary conditions.
        """
        self._ensure_state(SolverState.BOUNDARY_APPLIED)

        self.nodal_displacements = np.linalg.solve(
            self.global_stiffness_matrix,
            self.global_force_vector,
        )

        print("\n- Nodal displacements U:")
        print(self.nodal_displacements)

        self.state = SolverState.SOLVED

        return self.nodal_displacements, self.original_global_stiffness_matrix
