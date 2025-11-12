#!/usr/bin/env python
"""
Module defining the FEA solvers.

Created: 2025/10/18 10:24:33
Last modified: 2025/11/11 22:08:43
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import time
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from .fem import assemble_global_stiffness_matrix
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

    def __init__(self, model: Model, use_sparse: bool = True):
        """Initialize solver from a Model.

        Args:
            model: Model object containing mesh, element properties, BCs, and DOF space
            use_sparse: Use sparse matrix storage (default: True)

        Example:
            problem = Problem(Physics.MECHANICS, Dimension.D1)
            model = Model(mesh, problem)
            model.set_element_properties(element_properties)
            model.boundary_conditions.prescribe_displacement(...)

            solver = LinearStaticSolver(model)
        """
        self.mesh = model.mesh
        self.element_properties = model.element_properties

        # Registry-based BC system
        self.bc = model.bc
        self.registry = model.bc.registry

        self.dof_space = model.dof_space
        self.use_sparse = use_sparse
        self.state = SolverState.INITIALIZED

        # Initialize global matrices and vectors as instance attributes
        total_dofs = self.dof_space.total_dofs

        # Initialize empty structures
        if use_sparse:
            self.global_stiffness_matrix = sparse.csc_matrix((total_dofs, total_dofs))
        else:
            self.global_stiffness_matrix = np.zeros((total_dofs, total_dofs))

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
        self.solve_time: float = 0.0

    def _ensure_state(self, expected: SolverState) -> None:
        if self.state != expected:
            raise RuntimeError(
                f"Invalid solver state: expected {expected.name}, got {self.state.name}"
            )

    def assemble_global_matrix(self) -> None:
        """Constructs the system of equations KU=F."""

        self._ensure_state(SolverState.INITIALIZED)

        # Assemble the global stiffness matrix
        self.global_stiffness_matrix = assemble_global_stiffness_matrix(
            self.mesh,
            self.element_properties,
            self.global_stiffness_matrix,
            self.dof_space,
            use_sparse=self.use_sparse,
        )
        # print("\n- Global stiffness matrix K:")
        # for row in self.global_stiffness_matrix:
        #     print(row)

        self.state = SolverState.ASSEMBLED
        return None

    def apply_boundary_conditions(self) -> None:
        """Apply nodal forces from registry to global RHS vector."""

        self._ensure_state(SolverState.ASSEMBLED)

        # Pull Neumann forces from the registry
        neumann_forces = self.registry.get_neumann_forces()

        for dof, value in neumann_forces.items():
            self.global_force_vector[dof] += value

        self.state = SolverState.BOUNDARY_APPLIED
        return None

    def solve(self) -> None:
        """Solves the linear system KU=F using static condensation."""

        self._ensure_state(SolverState.BOUNDARY_APPLIED)

        K = self.global_stiffness_matrix
        F = self.global_force_vector

        # Extract prescribed DOF information
        dirichlet = self.registry.get_dirichlet_values()
        prescribed_dofs = np.array(list(dirichlet.keys()), dtype=int)
        prescribed_vals = np.array(list(dirichlet.values()), dtype=float)

        # Identify free DOFs
        all_dofs = np.arange(K.shape[0])
        free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

        # Partition the system
        if len(prescribed_dofs) > 0:
            K_ff = K[np.ix_(free_dofs, free_dofs)]
            K_fp = K[np.ix_(free_dofs, prescribed_dofs)]
            F_f = F[free_dofs] - K_fp @ prescribed_vals
        else:
            K_ff = K
            F_f = F[free_dofs]

        start_time = time.time()

        if self.use_sparse and sparse.isspmatrix(K):
            # Sparse solver with static condensation
            U_free = sparse.linalg.spsolve(K_ff, F_f)
        else:
            # Dense solver with static condensation
            U_free = np.linalg.solve(K_ff, F_f)

        self.solve_time = time.time() - start_time

        # Reconstruct full displacement vector
        self.nodal_displacements = np.zeros(K.shape[0])
        self.nodal_displacements[free_dofs] = U_free
        if len(prescribed_dofs) > 0:
            self.nodal_displacements[prescribed_dofs] = prescribed_vals

        # print("\n- Nodal displacements U:")
        # print(self.nodal_displacements)

        self._compute_statistics(K, free_dofs, prescribed_dofs)

        self.state = SolverState.SOLVED

        return None

    def _compute_statistics(self, K, free_dofs, prescribed_dofs) -> None:
        """Computes solver statistics."""

        if self.use_sparse and sparse.isspmatrix(K):
            # Total number of matrix entries
            self.num_matrix_entries = K.shape[0] * K.shape[1]

            # Matrix size in memory (bytes)
            self.matrix_size_bytes = K.data.nbytes + K.indices.nbytes + K.indptr.nbytes

            # Sparsity statistics
            self.num_nonzero_entries = K.nnz  # Number of stored values, includes zeros
            self.sparsity_percentage = (
                1.0 - self.num_nonzero_entries / self.num_matrix_entries
            ) * 100.0
        else:
            # Total number of matrix entries
            self.num_matrix_entries = self.global_stiffness_matrix.size

            # Matrix size in memory (bytes)
            self.matrix_size_bytes = self.global_stiffness_matrix.nbytes

            # Sparsity statistics
            self.num_nonzero_entries = int(
                np.count_nonzero(self.global_stiffness_matrix)
            )
            self.sparsity_percentage = (
                1.0 - self.num_nonzero_entries / self.num_matrix_entries
            ) * 100.0

        # Common statistics

        # System size (number of equations/unknowns)
        self.system_size = self.dof_space.total_dofs
        # Matrix dimensions (system_size x system_size)
        self.matrix_shape = self.global_stiffness_matrix.shape

        print(f"\n{'=' * 70}")
        print("Solver Statistics")
        print(f"{'=' * 70}")
        print(
            f"  Solver type:                  {'SPARSE' if self.use_sparse else 'DENSE'}"
        )
        print(f"  System size (DOFs):           {self.system_size}")
        print(f"  Free DOFs:                    {len(free_dofs):,}")
        print(f"  Prescribed DOFs:              {len(prescribed_dofs):,}")
        print(
            f"  Matrix shape:                 {self.matrix_shape[0]} x {self.matrix_shape[1]}"
        )
        print(f"  Total matrix entries:         {self.num_matrix_entries:,}")
        print(f"  Non-zero entries:             {self.num_nonzero_entries:,}")
        print(f"  Sparsity (% zeros):           {self.sparsity_percentage:.2f}%")
        print(
            f"  Matrix memory usage:          {self.matrix_size_bytes:,} bytes ({self.matrix_size_bytes / 1024 / 1024:.2f} MiB)"
        )
        print(f"  Solution time:                {self.solve_time:.4f} seconds")
        print(
            "  Note:                         Statistics for ORIGINAL matrix (before BCs)"
        )
        print(f"{'=' * 70}")

        plt.spy(K)

        return None
