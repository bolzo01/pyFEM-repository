#!/usr/bin/env python
"""
Module defining the BoundaryConditions class.

Created: 2025/10/25 19:28:51
Last modified: 2025/11/08 16:47:30
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from .dof_constraint_registry import DOFConstraintRegistry
from .dof_types import DOFSpace, DOFType
from .mesh import Mesh


class BoundaryConditions:
    """Manages boundary conditions for finite element analysis.

    Handles both Dirichlet (prescribed displacements) and Neumann (applied forces)
    boundary conditions. Supports specifying conditions on individual nodes or
    node sets.

    Users call methods like `prescribe_displacement()` or `apply_force()`.
    Internally, this class resolves node sets, maps them to global DOF indices,
    and delegates the actual DOF bookkeeping and conflict-prevention to the
    DOFConstraintRegistry.
    """

    def __init__(self, dof_space: DOFSpace, mesh: Mesh):
        """Initialize boundary conditions."""
        self.dof_space = dof_space
        self.mesh = mesh

        # Central low-level storage for DOF constraints
        self.registry = DOFConstraintRegistry()

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
            self.registry.set_dirichlet_value(global_dof, value)

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
            bc.apply_force((1, DOFType.U_X, 10.0)  # by tag (if it's a node set)
        """
        node_ids = self._resolve_nodes(nodes)

        for node in node_ids:
            global_dof = self.dof_space.get_global_dof(node, dof_type)
            self.registry.add_neumann_force(global_dof, value)

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
