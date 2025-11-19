#!/usr/bin/env python
"""
Module defining the FEA solvers.

Created: 2025/10/18 10:24:33
Last modified: 2025/11/14 01:23:28
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import time
from enum import Enum, auto

import numpy as np
from scipy import sparse

from .fem import assemble_global_stiffness_matrix
from .model import Model
from .solution import Solution


class SolverState(Enum):
    INITIALIZED = auto()
    ASSEMBLED = auto()
    BOUNDARY_APPLIED = auto()
    SOLVED = auto()


class LinearStaticSolver:
    """Solves linear static finite element problems.

    Assembles and solves the system of equations KU=F.

    Usage:
        solver = LinearStaticSolver(model)
        solver.assemble_global_matrix()
        solver.apply_boundary_conditions()
        solution = solver.solve()  # Returns Solution object
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
            model.bc.prescribe_displacement(...)

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

        # Performance metrics (populated during solve)
        self._solver_stats: dict[str, int | float] = {}

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

    def solve(self) -> Solution:
        """Solves the linear system KU=F using static condensation.

        Returns:
            Solution object containing nodal displacements and solver statistics
        """

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

        solve_time = time.time() - start_time

        # Reconstruct full displacement vector
        nodal_displacements = np.zeros(K.shape[0])
        nodal_displacements[free_dofs] = U_free
        if len(prescribed_dofs) > 0:
            nodal_displacements[prescribed_dofs] = prescribed_vals

        # Compute statistics
        self._compute_statistics(K, free_dofs, prescribed_dofs, solve_time)

        self.state = SolverState.SOLVED

        # Create and return Solution object
        solution = Solution(
            nodal_displacements=nodal_displacements,
            solver_stats=self._solver_stats.copy(),
        )

        return solution

    def _compute_statistics(
        self, K, free_dofs, prescribed_dofs, solve_time: float
    ) -> None:
        """Computes solver statistics and stores them in _solver_stats."""

        if self.use_sparse and sparse.isspmatrix(K):
            # Total number of matrix entries
            num_matrix_entries = K.shape[0] * K.shape[1]

            # Matrix size in memory (bytes)
            matrix_size_bytes = K.data.nbytes + K.indices.nbytes + K.indptr.nbytes

            # Sparsity statistics
            num_nonzero_entries = K.nnz  # Number of stored values, includes zeros
            sparsity_percentage = (
                1.0 - num_nonzero_entries / num_matrix_entries
            ) * 100.0
        else:
            # Total number of matrix entries
            num_matrix_entries = self.global_stiffness_matrix.size

            # Matrix size in memory (bytes)
            matrix_size_bytes = self.global_stiffness_matrix.nbytes

            # Sparsity statistics
            num_nonzero_entries = int(np.count_nonzero(self.global_stiffness_matrix))
            sparsity_percentage = (
                1.0 - num_nonzero_entries / num_matrix_entries
            ) * 100.0

        # Store statistics in dictionary
        self._solver_stats = {
            "system_size": self.dof_space.total_dofs,
            "free_dofs": len(free_dofs),
            "prescribed_dofs": len(prescribed_dofs),
            "matrix_shape_rows": self.global_stiffness_matrix.shape[0],
            "matrix_shape_cols": self.global_stiffness_matrix.shape[1],
            "total_matrix_entries": num_matrix_entries,
            "nonzero_entries": num_nonzero_entries,
            "sparsity_percentage": sparsity_percentage,
            "matrix_size_bytes": matrix_size_bytes,
            "solve_time": solve_time,
            "use_sparse": self.use_sparse,
        }

        # Print statistics
        print(f"\n{'=' * 70}")
        print("Solver Statistics")
        print(f"{'=' * 70}")
        print(
            f"  Solver type:                  {'SPARSE' if self.use_sparse else 'DENSE'}"
        )
        print(f"  System size (DOFs):           {self._solver_stats['system_size']}")
        print(f"  Free DOFs:                    {self._solver_stats['free_dofs']:,}")
        print(
            f"  Prescribed DOFs:              {self._solver_stats['prescribed_dofs']:,}"
        )
        print(
            f"  Matrix shape:                 {self._solver_stats['matrix_shape_rows']} x {self._solver_stats['matrix_shape_cols']}"
        )
        print(
            f"  Total matrix entries:         {self._solver_stats['total_matrix_entries']:,}"
        )
        print(
            f"  Non-zero entries:             {self._solver_stats['nonzero_entries']:,}"
        )
        print(
            f"  Sparsity (% zeros):           {self._solver_stats['sparsity_percentage']:.2f}%"
        )
        print(
            f"  Matrix memory usage:          {self._solver_stats['matrix_size_bytes']:,} bytes ({self._solver_stats['matrix_size_bytes'] / 1024 / 1024:.2f} MiB)"
        )
        print(
            f"  Solution time:                {self._solver_stats['solve_time']:.4f} seconds"
        )
        print(
            "  Note:                         Statistics for ORIGINAL matrix (before BCs)"
        )
        print(f"{'=' * 70}")

        return None
