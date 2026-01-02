#!/usr/bin/env python
"""
Module defining the BoundaryConditions class.

Created: 2025/10/25 19:28:51
Last modified: 2025/12/09 02:27:48
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from collections.abc import Callable

import numpy as np

from .dof_constraint_registry import DOFConstraintRegistry
from .dof_types import DOFSpace, DOFType
from .mesh import Mesh


class BoundaryConditions:
    """Manages boundary conditions for finite element analysis.

    Handles both Dirichlet (prescribed displacements) and Neumann (applied forces)
    boundary conditions. Supports specifying conditions on:
    - Individual nodes or node sets (by ID, name, or tag)
    - Node coordinates using predicates (geometric conditions)

    Users call methods like `prescribe_displacement()` or `apply_force()`.
    Internally, this class resolves node sets, maps them to global DOF indices,
    and delegates the actual DOF bookkeeping and conflict-prevention to the
    DOFConstraintRegistry.
    """

    def __init__(self, dof_space: DOFSpace, mesh: Mesh):
        """Initialize boundary conditions."""
        self.dof_space = dof_space
        self.mesh = mesh
        # store full prescribed values
        self._reference_displacements: dict[int, float] = {}

        # Central low-level storage for DOF constraints
        self.registry = DOFConstraintRegistry()

    # -------------------------------------------------------------------------
    # Standard boundary condition methods (node-based)
    # -------------------------------------------------------------------------

    def prescribe_displacement(
        self, nodes: int | set[int] | str, dof_type: DOFType, value: float
    ) -> None:
        """Apply a Dirichlet boundary condition (prescribed displacement).

        Args:
            nodes: Single node ID, set of node IDs, or node set name/tag
            dof_type: DOF type to constrain
            value: Prescribed value

        Example:
            bc.prescribe_displacement(0, DOFType.U_X, 0.0)  # single node
            bc.prescribe_displacement({0, 1}, DOFType.U_X, 0.0)  # set of nodes
            bc.prescribe_displacement("left_boundary", DOFType.U_X, 0.0)  # by name
            bc.prescribe_displacement(1, DOFType.U_X, 0.0)  # by tag (if it's a node set)
        """
        node_ids = self._resolve_nodes(nodes)

        for node in node_ids:
            global_dof = self.dof_space.get_global_dof(node, dof_type)

            # Store full (unscaled) value once for increments
            if global_dof not in self._reference_displacements:
                self._reference_displacements[global_dof] = value

            # Normal BC registration (used for non-incremental solve)
            self.registry.set_dirichlet_value(global_dof, value)

    def prescribe_dirichlet(
        self,
        node_set: str | int,
        dof_type: DOFType,
        value: float,
    ) -> None:
        """
        Apply a generic Dirichlet boundary condition on the given DOF type.

        This is a physics-agnostic entry point:
            - For mechanics: dof_type = U_X, U_Y, U_Z
            - For heat transfer: dof_type = TEMPERATURE
            - For other physics: corresponding primary DOF type

        Internally this forwards to the existing displacement-based
        infrastructure, which is already implemented in terms of
        (node_set, dof_type, value).
        """
        # Reuse existing implementation
        self.prescribe_displacement(node_set, dof_type, value)

    def apply_force(
        self, nodes: int | set[int] | str, dof_type: DOFType, value: float
    ) -> None:
        """Apply a Neumann boundary condition (nodal force).

        Args:
            nodes: Single node ID, set of node IDs, or node set name/tag
            dof_type: DOF type to apply force to
            value: Force value

        Example:
            bc.apply_force(3, DOFType.U_X, 10.0)  # single node
            bc.apply_force({3, 4}, DOFType.U_X, 5.0)  # set of nodes
            bc.apply_force("right_boundary", DOFType.U_X, 10.0)  # by name
            bc.apply_force(1, DOFType.U_X, 10.0)  # by tag (if it's a node set)
        """
        node_ids = self._resolve_nodes(nodes)

        for node in node_ids:
            global_dof = self.dof_space.get_global_dof(node, dof_type)
            self.registry.add_neumann_force(global_dof, value)

    def scale_displacements(self, load_factor: float):
        """
        Scale originally prescribed displacement BCs by a factor λ.
        """

        # nothing to scale if no reference values exist
        if not hasattr(self, "_reference_displacements"):
            return

        for dof, full_value in self._reference_displacements.items():
            scaled_value = load_factor * full_value
            self.registry.update_dirichlet_value(dof, scaled_value)

    # -------------------------------------------------------------------------
    # Coordinate-based boundary conditions
    # -------------------------------------------------------------------------

    def prescribe_where(
        self,
        predicate: Callable[[np.ndarray], bool],
        dof_type: DOFType,
        value: float,
        tol_factor: float = 1e-9,
    ) -> None:
        """
        Prescribe displacement based on node coordinates using a predicate function.

        The predicate receives a coordinate vector (ndarray) and returns True if
        the boundary condition should be applied at that location. Automatic
        floating-point tolerance is applied to handle numerical precision issues.

        Args:
            predicate: Function that takes coordinates and returns bool
            dof_type: DOF type to constrain
            value: Prescribed value
            tol_factor: Relative tolerance factor for coordinate matching

        Examples:
            # Fix nodes at x = 0
            model.bc.prescribe_where(lambda x: x[0] == 0, DOFType.U_X, 0.0)

            # Fix nodes at left edge (x < 0.01)
            model.bc.prescribe_where(lambda x: x[0] < 0.01, DOFType.U_X, 0.0)

            # Fix corner node at (0, 0)
            model.bc.prescribe_where(
                lambda x: x[0] == 0 and x[1] == 0,
                DOFType.U_Y,
                0.0
            )

            # Fix bottom edge
            model.bc.prescribe_where(lambda x: x[1] == 0, DOFType.U_Y, 0.0)
        """
        coords = self.mesh.points

        # Use mesh size to compute tolerance (automatic scale)
        bbox_size = np.max(coords) - np.min(coords)
        tol = tol_factor * max(bbox_size, 1.0)

        # Find all nodes that satisfy the predicate
        node_ids: list[int] = []
        for node_id, x in enumerate(coords):
            try:
                # Try predicate directly first
                if predicate(x):
                    node_ids.append(node_id)
                # If false, try with tolerance for floating-point robustness
                elif self._coord_match_predicate(x, predicate, tol):
                    node_ids.append(node_id)
            except Exception:
                # Predicate might fail for some coordinates (e.g., out of bounds)
                continue

        # Apply BC to all matching nodes
        for node_id in node_ids:
            self._assign_single_dof(node_id, dof_type, value)

    def apply_force_where(
        self,
        predicate: Callable[[np.ndarray], bool],
        dof_type: DOFType,
        value: float,
        tol_factor: float = 1e-9,
    ) -> None:
        """
        Apply force based on node coordinates using a predicate function.

        Args:
            predicate: Function that takes coordinates and returns bool
            dof_type: DOF type to apply force to
            value: Force value
            tol_factor: Relative tolerance factor for coordinate matching

        Example:
            # Apply load to right edge (x = L)
            L = 10.0
            model.bc.apply_force_where(
                lambda x: x[0] == L,
                DOFType.U_X,
                1000.0
            )
        """
        coords = self.mesh.points

        # Use mesh size to compute tolerance
        bbox_size = np.max(coords) - np.min(coords)
        tol = tol_factor * max(bbox_size, 1.0)

        # Find all nodes that satisfy the predicate
        node_ids: list[int] = []
        for node_id, x in enumerate(coords):
            try:
                if predicate(x) or self._coord_match_predicate(x, predicate, tol):
                    node_ids.append(node_id)
            except Exception:
                continue

        # Apply force to all matching nodes
        for node_id in node_ids:
            global_dof = self.dof_space.get_global_dof(node_id, dof_type)
            self.registry.add_neumann_force(global_dof, value)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _resolve_nodes(self, nodes: int | set[int] | str) -> set[int]:
        """Resolve nodes specification to a set of node IDs.

        Args:
            nodes: Single node ID, set of node IDs, or node set name

        Returns:
            Set of node IDs
        """
        if isinstance(nodes, int):
            # Check if it's a node set tag or a single node ID
            if nodes in self.mesh.node_sets:
                # It's a node set tag
                return self.mesh.node_sets[nodes].nodes
            else:
                # It's a single node ID
                return {nodes}
        elif isinstance(nodes, set):
            # Already a set of node IDs
            return nodes
        elif isinstance(nodes, str):
            # Node set name
            node_set = self.mesh.get_node_set(nodes)
            return node_set.nodes
        else:
            raise TypeError(f"nodes must be int, set[int], or str, got {type(nodes)}")

    def _coord_match_predicate(
        self,
        x: np.ndarray,
        predicate: Callable[[np.ndarray], bool],
        tol: float,
    ) -> bool:
        """
        Evaluate predicate robustly with tolerance for floating-point errors.

        If the predicate returns False, but would return True with a small
        numerical perturbation, treat it as True. This handles cases where
        nodes are "almost" at the specified coordinate due to mesh generation
        or floating-point arithmetic.

        Args:
            x: Node coordinates
            predicate: Coordinate test function
            tol: Tolerance for coordinate perturbation

        Returns:
            True if predicate matches within tolerance
        """
        # Already checked direct evaluation before calling this
        # Try small perturbations to catch numerical edge cases
        for delta in (-tol, tol):
            x_shift = x + delta
            try:
                if predicate(x_shift):
                    return True
            except Exception:
                pass
        return False

    def _assign_single_dof(self, node: int, dof_type: DOFType, value: float) -> None:
        """
        Assign a Dirichlet BC to a single node's DOF with conflict checking.

        Prevents assigning conflicting values to the same DOF. If the same
        value is assigned multiple times (e.g., from overlapping geometric
        conditions), it is silently accepted.

        Args:
            node: Node ID
            dof_type: DOF type
            value: Prescribed value

        Raises:
            ValueError: If attempting to prescribe a different value to an
                       already-constrained DOF
        """
        global_dof = self.dof_space.get_global_dof(node, dof_type)

        # Check if this DOF already has a Dirichlet condition
        existing_values = self.registry.get_dirichlet_values()
        if global_dof in existing_values:
            existing_value = existing_values[global_dof]
            if abs(existing_value - value) > 1e-12:
                raise ValueError(
                    f"Conflicting BC: node {node} DOF {dof_type.name} "
                    f"already prescribed to {existing_value}, cannot assign {value}"
                )
            # Same value - OK, ignore duplicate
            return

        # New prescription - assign it
        self.registry.set_dirichlet_value(global_dof, value)

    def print_summary(self):
        """Pretty print all boundary conditions (Dirichlet, Neumann)."""

        print("\n================= BOUNDARY CONDITIONS =================")

        registry = self.registry
        dof_space = self.dof_space

        # ---- Infer ndofs_per_node safely ----
        # Example: [UX, UY, UX, UY, UX, UY] → 2 DOFs per node
        global_types = dof_space.global_dof_types

        # Count how many DOFs until type sequence repeats → block size
        first_type = global_types[0]
        ndofs_per_node = 1
        for t in global_types[1:]:
            if t == first_type:
                break
            ndofs_per_node += 1

        # ---- Helpers ----
        def get_node_from_dof(dof: int) -> int:
            return dof // ndofs_per_node

        def get_type_from_dof(dof: int):
            return global_types[dof]

        # ---- DIRICHLET ----
        dirichlet = registry.get_dirichlet_values()
        print("\n Prescribed Displacements:")
        if dirichlet:
            for dof, value in sorted(dirichlet.items()):
                node = get_node_from_dof(dof)
                doftype = get_type_from_dof(dof).name
                print(f"   - Node {node:4d}, {doftype:6s} = {value:g}")
        else:
            print("   None")

        # ---- NEUMANN ----
        neumann = registry.get_neumann_forces()
        print("\n Applied Forces:")
        if neumann:
            for dof, value in sorted(neumann.items()):
                node = get_node_from_dof(dof)
                doftype = get_type_from_dof(dof).name
                print(f"   - Node {node:4d}, {doftype:6s} = {value:g}")
        else:
            print("   None")

        print("========================================================\n")
