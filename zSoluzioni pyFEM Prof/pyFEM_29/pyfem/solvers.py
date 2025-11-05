#!/usr/bin/env python
"""
Module defining the FEA solvers.

Created: 2025/10/18 10:24:33
Last modified: 2025/11/03 09:18:36
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from enum import Enum, auto

import numpy as np

from .fem import (
    apply_nodal_forces,
    apply_prescribed_displacements,
    assemble_global_stiffness_matrix,
)
from .model import Model


class SolverState(Enum):
    INITIALIZED = auto()
    ASSEMBLED = auto()
    BOUNDARY_APPLIED = auto()
    SOLVED = auto()


class LinearStaticSolver:
    """Solves linear static finite element problems.

    Assembles and solves the system of equations KU=F.
    """

    def __init__(self, model: Model):
        """Initialize solver from a Model.

        Args:
            model: Model object containing mesh, element properties, BCs, and DOF space

        Example:
            problem = Problem(Physics.MECHANICS, Dimension.D1)
            model = Model(mesh, problem)
            model.set_element_properties(element_properties)
            model.boundary_conditions.prescribe_displacement(...)

            solver = LinearStaticSolver(model)
        """
        self.mesh = model.mesh
        self.element_properties = model.element_properties
        self.applied_forces = model.bc.applied_forces
        self.prescribed_displacements = model.bc.prescribed_displacements
        self.dof_space = model.dof_space
        self.state = SolverState.INITIALIZED

        # Initialize global matrices and vectors as instance attributes
        total_dofs = self.dof_space.total_dofs
        self.global_stiffness_matrix = np.zeros((total_dofs, total_dofs))
        self.original_global_stiffness_matrix = np.zeros((total_dofs, total_dofs))
        self.global_force_vector = np.zeros(total_dofs)

        # Will be computed by solve()
        self.nodal_displacements: np.ndarray

        # Performance metrics
        self.system_size: int = 0
        self.matrix_shape: tuple[int, int] = (0, 0)
        self.num_matrix_entries: int = 0
        self.matrix_size_bytes: int = 0
        self.num_nonzero_entries: int = 0
        self.sparsity_percentage: float = 0.0

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
            self.mesh,
            self.element_properties,
            self.global_stiffness_matrix,
            self.dof_space,
        )
        # print("\n- Global stiffness matrix K:")
        # for row in self.global_stiffness_matrix:
        #     print(row)

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

        # print(
        #     "\n- Modified global stiffness matrix K after applying boundary conditions:"
        # )
        # for row in self.global_stiffness_matrix:
        #     print(row)

        # print("\n- Global force vector F after applying boundary conditions:")
        # print(self.global_force_vector)

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

        # System size (number of equations/unknowns)
        self.system_size = self.dof_space.total_dofs

        # Matrix dimensions (system_size × system_size)
        self.matrix_shape = self.global_stiffness_matrix.shape

        # Total number of matrix entries
        self.num_matrix_entries = self.global_stiffness_matrix.size

        # Matrix size in memory (bytes)
        self.matrix_size_bytes = self.global_stiffness_matrix.nbytes

        # Sparsity statistics
        self.num_nonzero_entries = int(np.count_nonzero(self.global_stiffness_matrix))
        self.sparsity_percentage = (
            1.0 - self.num_nonzero_entries / self.num_matrix_entries
        ) * 100.0

        print(f"\n{'=' * 70}")
        print("Solver Statistics")
        print(f"{'=' * 70}")
        print(f"  System size (DOFs):           {self.system_size}")
        print(
            f"  Matrix shape:                 {self.matrix_shape[0]} × {self.matrix_shape[1]}"
        )
        print(f"  Total matrix entries:         {self.num_matrix_entries:,}")
        print(f"  Non-zero entries:             {self.num_nonzero_entries:,}")
        print(f"  Sparsity (% zeros):           {self.sparsity_percentage:.2f}%")
        print(
            f"  Matrix memory usage:          {self.matrix_size_bytes:,} bytes ({self.matrix_size_bytes / 1024 / 1024:.2f} MiB)"
        )
        print(f"{'=' * 70}")

        # print("\n- Nodal displacements U:")
        # print(self.nodal_displacements)

        self.state = SolverState.SOLVED

        return self.nodal_displacements, self.original_global_stiffness_matrix
