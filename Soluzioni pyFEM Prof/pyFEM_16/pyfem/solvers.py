#!/usr/bin/env python
"""
Module defining the FEA solvers.

Created: 2025/10/18 10:24:33
Last modified: 2025/10/18 17:30:51
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from .fem import (
    apply_nodal_forces,
    apply_prescribed_displacements,
    assemble_global_stiffness_matrix,
)
from .materials import Materials
from .mesh import Mesh


class LinearStaticSolver:
    """Solves linear static finite element problems.

    Assembles and solves the system of equations KU=F.
    """

    def __init__(
        self,
        mesh: Mesh,
        materials: Materials,
        applied_forces: list[tuple[int, float]],
        prescribed_displacements: list[tuple[int, float]],
        dofs_per_node: int,
    ):
        self.mesh = mesh
        self.materials = materials
        self.applied_forces = applied_forces
        self.prescribed_displacements = prescribed_displacements
        self.dofs_per_node = dofs_per_node

        # Initialize global matrices and vectors as instance attributes
        total_dofs = self.dofs_per_node * self.mesh.num_nodes
        self.global_stiffness_matrix = np.zeros((total_dofs, total_dofs))
        self.original_global_stiffness_matrix = np.zeros((total_dofs, total_dofs))
        self.global_force_vector = np.zeros(total_dofs)

        # Will be computed by solve()
        self.nodal_displacements: np.ndarray

    def assemble_global_matrix(self) -> None:
        """Constructs the system of equations KU=F."""

        # Assemble the global stiffness matrix
        assemble_global_stiffness_matrix(
            self.mesh, self.materials, self.global_stiffness_matrix
        )
        print("\n- Global stiffness matrix K:")
        for row in self.global_stiffness_matrix:
            print(row)

        # Save a copy of the original global stiffness matrix before applying boundary conditions
        self.original_global_stiffness_matrix = self.global_stiffness_matrix.copy()

        return None

    def apply_boundary_conditions(self) -> None:
        """Applies Neumann and Dirichlet boundary conditions."""

        # Compute total number of DOFs
        total_dofs = self.dofs_per_node * self.mesh.num_nodes

        # Boundary conditions: Apply forces
        apply_nodal_forces(self.applied_forces, self.global_force_vector)

        # Boundary conditions: Constrain displacements
        apply_prescribed_displacements(
            self.prescribed_displacements,
            self.global_stiffness_matrix,
            self.global_force_vector,
            total_dofs,
        )

        print(
            "\n- Modified global stiffness matrix K after applying boundary conditions:"
        )
        for row in self.global_stiffness_matrix:
            print(row)

        print("\n- Global force vector F after applying boundary conditions:")
        print(self.global_force_vector)

        return None

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """Solves the linear system of equations KU=F.

        Returns:
            Solution array and global stiffness matrix before applying boundary conditions.
        """

        self.nodal_displacements = np.linalg.solve(
            self.global_stiffness_matrix,
            self.global_force_vector,
        )

        print("\n- Nodal displacements U:")
        print(self.nodal_displacements)

        return self.nodal_displacements, self.original_global_stiffness_matrix
