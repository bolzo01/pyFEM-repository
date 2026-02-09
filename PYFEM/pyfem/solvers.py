#!/usr/bin/env python
"""
Module defining the FEA solvers.

Created: 2025/10/18 10:24:33
Last modified: 2026/02/08 18:00:47
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import time
from enum import Enum, auto

import numpy as np
from scipy import sparse

from .dof_types import DOFType
from .fem import assemble_global_stiffness_matrix
from .model import Model
from .problem import Physics
from .solution import Solution


class SolverState(Enum):
    INITIALIZED = auto()
    ASSEMBLED = auto()
    BOUNDARY_APPLIED = auto()
    SOLVED = auto()


class LinearStaticSolver:
    """
    Solves linear static finite element problems (KU = F).

    Workflow:
        assemble_global_matrix()
        apply_boundary_conditions()
        solution = solve()

    Usage:
        solver = LinearStaticSolver(model)
        solver.assemble_global_matrix()
        solver.apply_boundary_conditions()
        solution = solver.solve()  # Returns Solution object with reaction forces

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
        self.model = model
        self.mesh = model.mesh
        self.element_properties = model.element_properties
        # Registry-based BC system
        self.bc = model.bc  # registry wrapper
        self.registry = model.bc.registry  # actual constraint registry
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

        # Solver statistics (populated in solve)
        self._solver_stats: dict[str, int | float] = {}

    # ------------------------------------------------------------
    # State machine helper
    # ------------------------------------------------------------
    def _ensure_state(self, expected: SolverState) -> None:
        if self.state != expected:
            raise RuntimeError(
                f"Solver in invalid state: expected {expected.name}, got {self.state.name}"
            )

    # ------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------
    def assemble_global_matrix(self) -> None:
        """Constructs the system of equations KU=F."""
        self._ensure_state(SolverState.INITIALIZED)

        # Assemble the global stiffness matrix
        self.global_stiffness_matrix = assemble_global_stiffness_matrix(
            mesh=self.mesh,
            element_properties=self.element_properties,
            materials=self.model.materials,
            global_stiffness_matrix=self.global_stiffness_matrix,
            dof_space=self.dof_space,
            use_sparse=self.use_sparse,
        )
        # print("\n- Global stiffness matrix K:")
        # for row in self.global_stiffness_matrix:
        #     print(row)
        self.state = SolverState.ASSEMBLED
        return None

    # ------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------
    def apply_boundary_conditions(self, load_factor=1.0):
        """Apply nodal forces from registry to global RHS vector."""
        self._ensure_state(SolverState.ASSEMBLED)

        # Pull Neumann forces from the registry
        neumann_forces = self.registry.get_neumann_forces()

        for dof, value in neumann_forces.items():
            self.global_force_vector[dof] += load_factor * value

        self.state = SolverState.BOUNDARY_APPLIED
        return None

    # ------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------
    def solve(self, compute_reactions: bool = True) -> Solution:
        """Solves the linear system KU = F by partitioning into free and prescribed DOFs (static condensation).

        Args:
            compute_reactions: Whether to compute reaction forces (default: True)

        Returns:
            Solution object containing nodal displacements, reactions, and statistics
        """
        self._ensure_state(SolverState.BOUNDARY_APPLIED)

        K = self.global_stiffness_matrix
        F = self.global_force_vector

        # Extract Dirichlet info
        dirichlet = self.registry.get_dirichlet_values()
        prescribed_dofs = np.array(list(dirichlet.keys()), dtype=int)
        prescribed_vals = np.array(list(dirichlet.values()), dtype=float)

        # Identify free DOFs
        all_dofs = np.arange(self.dof_space.total_dofs)
        free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)

        # Partition the system
        if prescribed_dofs.size > 0:
            K_ff = K[np.ix_(free_dofs, free_dofs)]
            K_fp = K[np.ix_(free_dofs, prescribed_dofs)]
            F_f = F[free_dofs] - K_fp @ prescribed_vals
        else:
            K_ff = K
            F_f = F[free_dofs]

        # Solve system
        start = time.time()

        if self.use_sparse and sparse.isspmatrix(K):
            U_free = sparse.linalg.spsolve(K_ff, F_f)
        else:
            U_free = np.linalg.solve(K_ff, F_f)

        solve_time = time.time() - start

        # Reconstruct global displacement vector
        U = np.zeros_like(F)
        U[free_dofs] = U_free
        if prescribed_dofs.size > 0:
            U[prescribed_dofs] = prescribed_vals

        # Compute reaction forces: R = K*u - f_applied
        reactions = None
        if compute_reactions:
            reactions = self._compute_reactions(K, U, F, prescribed_dofs)

        # Solver statistics
        self._compute_statistics(K, free_dofs, prescribed_dofs, solve_time)

        self.state = SolverState.SOLVED

        # Create and return Solution object
        solution = Solution(
            nodal_displacements=U,
            reaction_forces=reactions,
            dof_types=self.model.dof_space.global_dof_types,
            solver_stats=self._solver_stats,
        )

        return solution

    # ------------------------------------------------------------
    # Reaction forces
    # ------------------------------------------------------------
    def _compute_reactions(
        self,
        K: np.ndarray | sparse.spmatrix,
        U: np.ndarray,
        F: np.ndarray,
        prescribed_dofs: np.ndarray,
    ) -> np.ndarray:
        """Compute reaction forces at constrained DOFs.

        Reactions are computed as: R = K*u - f_applied

        At free DOFs: R = 0 (equilibrium is satisfied by construction)
        At prescribed DOFs: R != 0 (these are the reaction forces)

        Args:
            K: Global stiffness matrix
            u: Nodal displacements (full vector)
            f_applied: Applied force vector (full vector)
            prescribed_dofs: Indices of prescribed DOFs

        Returns:
            Reaction force vector (same size as u)
        """

        # Reactions: R = K*u - f_applied
        R = K @ U - F
        free = np.setdiff1d(np.arange(len(U)), prescribed_dofs)

        # Sanity check
        tol = max(1e-12, 1e-8 * float(np.linalg.norm(K @ U)))
        if np.any(np.abs(R[free]) > tol):
            max_r = np.max(np.abs(R[free]))
            print(f"Warning: residual at free DOFs too large: {max_r:e}")

        return R

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------
    def _compute_statistics(
        self, K, free_dofs, prescribed_dofs, solve_time: float
    ) -> None:
        """Computes solver statistics and stores them in _solver_stats."""

        if self.use_sparse and sparse.isspmatrix(K):
            # Total number of matrix entries
            num_entries = K.shape[0] * K.shape[1]
            # Matrix size in memory (bytes)
            matrix_size_bytes = K.data.nbytes + K.indices.nbytes + K.indptr.nbytes
            # Number of stored values, includes zeros
            nonzeros = K.nnz
        else:
            # Total number of matrix entries
            num_entries = K.size
            # Matrix size in memory (bytes)
            matrix_size_bytes = K.nbytes
            # Number of non zero entries
            nonzeros = int(np.count_nonzero(K))

        sparsity = (1.0 - nonzeros / num_entries) * 100.0
        # Store statistics in dictionary
        self._solver_stats = {
            "system_size": self.dof_space.total_dofs,
            "free_dofs": len(free_dofs),
            "prescribed_dofs": len(prescribed_dofs),
            "total_matrix_entries": num_entries,
            "nonzero_entries": nonzeros,
            "sparsity_percentage": sparsity,
            "matrix_size_bytes": matrix_size_bytes,
            "solve_time": solve_time,
            "use_sparse": self.use_sparse,
        }

        # Print statistics
        print("\n" + "=" * 70)
        print("Solver Statistics")
        print("=" * 70)
        print(f"  System size (DOFs):           {self._solver_stats['system_size']}")
        print(f"  Free DOFs:                    {self._solver_stats['free_dofs']:,}")
        print(
            f"  Prescribed DOFs:              {self._solver_stats['prescribed_dofs']:,}"
        )
        print(
            f"  Non-zero entries:             {self._solver_stats['nonzero_entries']:,}"
        )
        print(
            f"  Sparsity (% zeros):           {self._solver_stats['sparsity_percentage']:.2f}%"
        )
        print(
            f"  Matrix memory usage:          {matrix_size_bytes / 1024 / 1024:.2f} MiB"
        )
        print(f"  Solve time:                   {solve_time:.4f} s")
        print("=" * 70)

        return None


class DiffusionExplicitSolver:
    """
    Explicit forward Euler solver for transient diffusion with stability checking.

    Stability condition for 1D heat equation:
        Δt ≤ Δx² / (2α)

    where α is thermal diffusivity and Δx is minimum element size.
    """

    def __init__(self, model):
        self.model = model
        self.mesh = model.mesh
        self.dof_space = model.dof_space
        self.element_properties = model.element_properties
        self.materials = model.materials
        self.M_inv = None
        self._sources_are_constant = None  # None = "Non ho ancora controllato"
        self._force_vector_is_ready = False

        # Enforce that this solver is used only for heat-transfer problems
        if self.model.problem.physics is not Physics.HEAT_TRANSFER:
            raise ValueError(
                "DiffusionExplicitSolver currently supports only "
                "Physics.HEAT_TRANSFER problems."
            )

        # Ensure temperature DOF is active and remember it explicitly
        if DOFType.TEMPERATURE not in self.dof_space.active_dof_types:
            raise ValueError(
                "DiffusionExplicitSolver requires DOFType.TEMPERATURE "
                "to be active in the model's DOF space."
            )

        self._temperature_dof_type = DOFType.TEMPERATURE

        ndofs = self.dof_space.total_dofs

        # Global matrices/vectors
        self.M_lumped = np.zeros(ndofs, dtype=float)
        self.K = np.zeros((ndofs, ndofs), dtype=float)
        self.F = np.zeros(ndofs, dtype=float)

        # Stability bookkeeping
        self._dt_critical = None
        self._min_element_size = None
        self._max_diffusivity = None

    def compute_critical_timestep(self) -> float:
        """
        Calculate the critical time step for stability.

        For forward Euler: Δt_crit = C * min(Δx²/α)
        where C = 0.5 for 1D (conservative: 0.4)

        Returns:
            Critical time step in seconds
        """

        min_dx_sq = float("inf")
        max_alpha = 0.0

        for e in range(self.mesh.num_elements):
            conn = self.mesh.element_connectivity[e]
            coords = self.mesh.points[conn]
            prop_label = self.mesh.element_property_labels[e]
            elem_prop = self.element_properties[prop_label]

            # Element size (works for 1D, extend for 2D/3D)
            if coords.ndim == 1:
                dx = abs(coords[1] - coords[0])
            else:
                # For multi-dimensional: minimum edge length
                dx = float("inf")
                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        dist = np.linalg.norm(coords[j] - coords[i])
                        dx = min(dx, dist)

            min_dx_sq = min(min_dx_sq, dx**2)

            # Get diffusivity from material
            if elem_prop.material is not None:
                material = self.materials[elem_prop.material]
                alpha = getattr(material, "alpha", None)
                if alpha is not None:
                    max_alpha = max(max_alpha, alpha)

        if max_alpha == 0.0:
            raise ValueError("No valid diffusivity found in materials")

        # Conservative safety factor for 1D: 0.4 instead of theoretical 0.5
        safety_factor = 0.9
        dt_crit = safety_factor * min_dx_sq / max_alpha

        # Store for reporting
        self._dt_critical = dt_crit
        self._min_element_size = np.sqrt(min_dx_sq)
        self._max_diffusivity = max_alpha

        return dt_crit

    def compute_critical_timestep_2D(self) -> float:
        """
        Calculate the critical time step for stability.

        For forward Euler: Δt_crit = min(m_i/K_ii)
        where: m_i = M_lumped[i]
               K_ii = K[i, i] = diagonal of stiffness matrix

        Returns:
            Critical time step in seconds
        """
        from scipy import sparse

        use_sparse = True
        self.M_lumped, self.K = self.assemble_mass_and_stiffness(use_sparse)

        if sparse.issparse(self.K):
            K_diag = self.K.diagonal()  # Sparse matrix method
        else:
            K_diag = np.diag(self.K)  # Dense matrix method (Numpy)

        # Avoid division by zero
        K_diag_safe = K_diag.copy()
        K_diag_safe[K_diag_safe < 1e-15] = 1e-15

        # dt_crit computation
        dt = self.M_lumped / K_diag_safe
        dt_crit_theorethical = float(np.min(dt))

        # Safety factor of 0.9
        dt_crit = dt_crit_theorethical * 0.9

        # Store for reporting
        self._dt_critical = dt_crit

        return dt_crit

    def check_stability(self, dt: float, verbose: bool = False) -> bool:
        """
        Check if the given time step satisfies stability condition.

        Args:
            dt: Proposed time step
            verbose: Print stability information

        Returns:
            True if stable, False otherwise
        """

        # We understand the mesh dimension from the points array dimension
        ndim = self.mesh.points.shape[1]

        # We compute dt_crit
        if ndim == 2:
            dt_crit = self.compute_critical_timestep_2D()

            is_stable = dt <= dt_crit
            ratio = dt / dt_crit

            if verbose:
                print("\n" + "=" * 70)
                print("STABILITY CHECK: Forward Euler Time Stepping")
                print("=" * 70)
                print(
                    f"  Critical time step:        {dt_crit:.3e} s ({dt_crit / 3600:.2f} hours)"
                )
                print(
                    f"  Requested time step:       {dt:.3e} s ({dt / 3600:.2f} hours)"
                )
                print(f"  Stability ratio (Δt/Δt_c): {ratio:.3f}")

                if is_stable:
                    if ratio > 0.8:
                        print("  Status: MARGINALLY STABLE (ratio > 0.8)")
                        print(
                            f"  Recommendation: Reduce Δt to {0.5 * dt_crit:.3e} s for safety"
                        )
                    else:
                        print("  Status: ✓ STABLE")
                else:
                    print("  Status: UNSTABLE - SOLUTION WILL DIVERGE!")
                    print(f"  Required: Δt ≤ {dt_crit:.3e} s")
                print("=" * 70)

        elif ndim == 1:
            dt_crit = self.compute_critical_timestep()

            is_stable = dt <= dt_crit
            ratio = dt / dt_crit

            if verbose:
                print("\n" + "=" * 70)
                print("STABILITY CHECK: Forward Euler Time Stepping")
                print("=" * 70)
                print(
                    f"  Critical time step:        {dt_crit:.3e} s ({dt_crit / 3600:.2f} hours)"
                )
                print(
                    f"  Requested time step:       {dt:.3e} s ({dt / 3600:.2f} hours)"
                )
                print(f"  Stability ratio (Δt/Δt_c): {ratio:.3f}")
                if is_stable:
                    if ratio > 0.8:
                        print("  Status: MARGINALLY STABLE (ratio > 0.8)")
                        print(
                            f"  Recommendation: Reduce Δt to {0.5 * dt_crit:.3e} s for safety"
                        )
                    else:
                        print("  Status: ✓ STABLE")
                else:
                    print("  Status: UNSTABLE - SOLUTION WILL DIVERGE!")
                    print(f"  Required: Δt ≤ {dt_crit:.3e} s")
                print("=" * 70)

        return is_stable

    def assemble_mass_and_stiffness(
        self, use_sparse: bool = True
    ) -> tuple[np.ndarray, np.ndarray | sparse.spmatrix]:
        """
        Assemble global lumped mass vector M_lumped and stiffness K
        using element registry pattern.
        """
        from scipy import sparse

        from .elements.element_registry import create_element

        # Reset Mass matrix
        self.M_lumped[:] = 0.0

        # Initialize stiffness matrix
        if use_sparse:
            # Dizionario per accumulare i valori sparsi: Key=(row, col), Value=stiffness
            K_triplets: dict[tuple[int, int], float] = {}
        else:
            # Se usiamo denso, resettiamo la matrice esistente
            # Nota: self.K deve essere stata inizializzata come densa in __init__
            if sparse.issparse(self.K):
                # Se era sparsa e ora vogliamo densa, dobbiamo ricrearla
                self.K = np.zeros(
                    (self.dof_space.total_dofs, self.dof_space.total_dofs)
                )
            else:
                self.K[:, :] = 0.0

        for e in range(self.mesh.num_elements):
            conn = self.mesh.element_connectivity[e]
            prop_label = self.mesh.element_property_labels[e]
            elem_prop = self.element_properties[prop_label]

            # Create element instance through registry
            element = create_element(elem_prop, e)
            coords = self.mesh.points[conn]

            # Get material
            material = None
            if elem_prop.material is not None:
                material = self.materials[elem_prop.material]

            # Check if element supports diffusion
            if not hasattr(element, "compute_diffusion_matrices"):
                raise NotImplementedError(
                    f"Element kind '{elem_prop.kind}' does not support "
                    f"diffusion analysis. Implement compute_diffusion_matrices()."
                )

            # Element mass and stiffness
            M_e, K_e = element.compute_diffusion_matrices(
                coords, material, elem_prop.params
            )

            temp_dof = self._temperature_dof_type
            edofs = [self.dof_space.get_global_dof(node, temp_dof) for node in conn]

            # Lump mass matrix (row-sum)
            for a_local, a_global in enumerate(edofs):
                self.M_lumped[a_global] += M_e[a_local, :].sum()

            # Assemble stiffness
            if use_sparse:
                # Accumulate into dictionary (sums duplicates manually)
                for a_local, a_global in enumerate(edofs):
                    for b_local, b_global in enumerate(edofs):
                        key = (a_global, b_global)
                        val = K_e[a_local, b_local]
                        if val != 0.0:  # Ignore pure zeros
                            K_triplets[key] = K_triplets.get(key, 0.0) + val
            else:
                # Metodo Denso Classico
                for a_local, a_global in enumerate(edofs):
                    for b_local, b_global in enumerate(edofs):
                        self.K[a_global, b_global] += K_e[a_local, b_local]

        if use_sparse:
            # Remove entries that cancel to ~0 due to numerical roundoff
            tolerance = 1e-13
            K_filtered_triplets = [
                (i, j, v) for (i, j), v in K_triplets.items() if abs(v) > tolerance
            ]

            # Setting up data for Scipy (COO format)
            n_entries = len(K_filtered_triplets)
            data = np.empty(n_entries, dtype=float)
            row = np.empty(n_entries, dtype=int)
            col = np.empty(n_entries, dtype=int)

            for idx, (i, j, value) in enumerate(K_filtered_triplets):
                row[idx] = i
                col[idx] = j
                data[idx] = value

            # Single conversion: COO -> CSC
            n_dof = self.dof_space.total_dofs
            self.K = sparse.csc_matrix((data, (row, col)), shape=(n_dof, n_dof))

        return self.M_lumped, self.K

    def assemble_force_vector(self, time: float = 0.0) -> None:
        """
        Assemble global force vector F from source terms.

        Args:
            time: Current simulation time (for time-dependent sources)
        """

        # If the sources are constant and we have already assembled F, we can skip re-assembly
        if self._sources_are_constant is True and self._force_vector_is_ready:
            return

        from .elements.element_registry import create_element

        # Reset global force vector
        self.F[:] = 0.0

        # Temporary flag to check if we encounter any time-dependent source term
        found_time_dependent_param = False

        for e in range(self.mesh.num_elements):
            conn = self.mesh.element_connectivity[e]
            prop_label = self.mesh.element_property_labels[e]
            elem_prop = self.element_properties[prop_label]

            # To control if the system has time-dependent sources
            if self._sources_are_constant is None:
                if elem_prop.params:
                    for value in elem_prop.params.values():
                        # If the value is a FUNCTION (callable), then it varies in time!
                        if callable(value):
                            found_time_dependent_param = True

            # Create element instance
            element = create_element(elem_prop, e)
            coords = self.mesh.points[conn]

            # Check if element has a source term implementation
            if not hasattr(element, "compute_source_vector"):
                # No source term for this element
                continue

            # Element source vector f_e
            f_e = element.compute_source_vector(coords, elem_prop.params, time)

            temp_dof = self._temperature_dof_type
            edofs = [self.dof_space.get_global_dof(node, temp_dof) for node in conn]

            for a_local, a_global in enumerate(edofs):
                self.F[a_global] += f_e[a_local]

        # Memorizing of the state of the sources (only after the first cicle)
        if self._sources_are_constant is None:
            if found_time_dependent_param:
                print(
                    "INFO: Time-dependent sources detected (functions). Optimization OFF."
                )
                self._sources_are_constant = False
            else:
                print("INFO: Static sources detected (numbers). Optimization ON.")
                self._sources_are_constant = True

        self._force_vector_is_ready = True

    def forward_euler_step(self, T_n: np.ndarray, dt: float, time: float) -> np.ndarray:
        """
        Compute T^{n+1} = T^n + dt * M_lumped^{-1} (F^n - K T^n).

        Args:
            T_n: Temperature at time level n
            dt: Time step
            time: Current time (for time-dependent forces)

        Returns:
            T^{n+1}: Temperature at next time level
        """
        # Update force vector if time-dependent
        self.assemble_force_vector(time)

        # Residual: r = F - K T^n
        r = self.F - self.K @ T_n

        # Avoid division by zero (Dirichlet DOFs)
        M_inv = np.zeros_like(self.M_lumped)
        nonzero = self.M_lumped > 1e-14
        M_inv[nonzero] = 1.0 / self.M_lumped[nonzero]

        if self.M_inv is None:
            # Create the blank vector
            self.M_inv = np.zeros_like(self.M_lumped)

            # Avoid division by zero (Dirichlet DOFs)
            nonzero = self.M_lumped > 1e-14

            # Inverse computation
            self.M_inv[nonzero] = 1.0 / self.M_lumped[nonzero]
            print("INFO: Mass matrix inverted and cached.")

        # Explicit update
        T_np1 = T_n + dt * (M_inv * r)

        return T_np1

    def apply_dirichlet_temperatures(self, T: np.ndarray, time: float) -> None:
        """
        Enforce Dirichlet temperature BCs.

        Args:
            T: Temperature vector to modify
            time: Current time (for time-dependent BCs)
        """
        dirichlet = self.model.bc.registry.get_dirichlet_values()

        for dof, value in dirichlet.items():
            # Support time-dependent BCs (callable values)
            if callable(value):
                T[dof] = value(time)
            else:
                T[dof] = value

    def initial_temperature(self) -> np.ndarray:
        """
        Return the initial temperature field T^0.

        Priority:
            1) If the model defines an initial temperature via
               model.set_initial_temperature, return that.
            2) Otherwise, default to zeros.
        """
        ndofs = self.dof_space.total_dofs

        if self.model.has_initial_temperature():
            return self.model.get_initial_temperature()
        return np.zeros(ndofs, dtype=float)

    def compute_heat_flux(
        self, T: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        """
        Computes the reaction forces (Heat Flux in Watts) at the boundaries.

        Physics:
            In steady state: K * T = F_source + F_reaction
            Therefore:       F_reaction = K * T - F_source

        Args:
            T (np.ndarray): The temperature vector (solution).
            verbose (bool): If True, prints a summary to the terminal.

        Returns:
                - 'reaction_vector': The nodal reaction forces [Watts].
                - 'power_in': Total power supplied TO the system (e.g., hot tubes) [Watts].
                - 'power_out': Total power removed FROM the system (e.g., cold walls) [Watts].
                - 'balance': Net balance (should be ~0 at steady state) [Watts].
        """

        # 3. Calculate Reaction Forces: F_reaction = K*T - F_internal
        #    K*T represents the internal resistance forces
        #    self.F represents the internal heat generation (F_source) (source terms)
        F_internal = self.K @ T
        F_reaction = F_internal - self.F

        # 4. Analyze Fluxes
        #    Positive values (> 0): Heat entering the domain (Hot source)
        #    Negative values (< 0): Heat leaving the domain (Cold sink)
        tol = 1e-9
        power_in = np.sum(F_reaction[F_reaction > tol])
        power_out = np.sum(F_reaction[F_reaction < -tol])

        # Net balance (Energy conservation check)
        balance = power_in + power_out

        return F_reaction, power_in, power_out, balance
