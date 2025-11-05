#!/usr/bin/env python
"""
Module defining DOF types and DOF space management.

Created: 2025/10/19 18:19:46
Last modified: 2025/10/29 00:05:41
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from enum import Enum, unique
from typing import Protocol

# Type alias for entity keys (tuples of ints and/or strings)
EntityKey = tuple[int | str, ...]


# -----------------------------------------------------------------------------
# Structural typing interface (Protocol)
# -----------------------------------------------------------------------------
class HasDOFTypes(Protocol):
    """Any object that can provide its associated DOF types."""

    def get_dof_types(self) -> list["DOFType"]: ...


# -----------------------------------------------------------------------------
# DOFType enumeration type
# -----------------------------------------------------------------------------
@unique
class DOFType(Enum):
    """Enumeration of degree of freedom types."""

    # Displacement vector components
    U_X = "u_x"
    U_Y = "u_y"
    U_Z = "u_z"

    # Others
    ELECTRIC_POTENTIAL = "electric_potential"
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"


# -----------------------------------------------------------------------------
# DOFSpace class
# -----------------------------------------------------------------------------
class DOFSpace:
    """Manages the mapping between entities, DOF types, and global DOF indices.

    The DOFSpace class is responsible for:
    - Declaring which DOF types are active in the problem
    - Assigning global DOF indices to (entity, DOF type) pairs
    - Providing lookup functionality to retrieve global DOF indices

    Entities are identified by tuples to support various use cases:
    - Mesh nodes: (node_id,)
    - Embedded nodes: (label, node_id)
    - GFEM/XFEM enriched DOFs: (node_id, enrichment_id)
    - Any other hierarchical identification scheme

    Attributes:
        active_dof_types: List of DOF types that are active in this problem
        dofs: Mapping from entity key to a dict of {DOF type: global DOF index}
        _next_global_dof: Counter for assigning new global DOF indices
    """

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.active_dof_types: list[DOFType] = []
        self.dofs: dict[EntityKey, dict[DOFType | str, int]] = {}
        self._next_global_dof = 0

    # -------------------------------------------------------------------------
    def activate_dof_types(self, *dof_types: DOFType) -> None:
        """Declare which DOF types will be used in this problem.

        Args:
            *dof_types: Variable number of DOFType enum values

        Example:
            dof_space.activate_dof_types(DOFType.U_X, DOFType.U_Y)
        """
        self.active_dof_types.extend(dof_types)

    # -------------------------------------------------------------------------
    def activate_dof_types_for_problem(self, problem: HasDOFTypes) -> None:
        """Activate DOF types based on a problem or any HasDOFTypes object."""
        dof_types = problem.get_dof_types()
        self.activate_dof_types(*dof_types)

    # -------------------------------------------------------------------------
    def assign_dofs_to_node(
        self, node: int, dof_types: list[DOFType] | None = None
    ) -> None:
        """Assign DOF types to a mesh node.

        Args:
            node: Node ID
            dof_types: List of DOF types to assign. If None, assigns all active types.

        Example:
            dof_space.assign_dofs_to_node(0, [DOFType.U_X])
            dof_space.assign_dofs_to_node(1)  # assigns all active types
        """
        if dof_types is None:
            dof_types = self.active_dof_types

        entity_key = (node,)
        if entity_key not in self.dofs:
            self.dofs[entity_key] = {}

        for dof_type in dof_types:
            if dof_type not in self.dofs[entity_key]:
                self.dofs[entity_key][dof_type] = self._next_global_dof
                self._next_global_dof += 1

    # -------------------------------------------------------------------------
    def assign_dofs_to_all_nodes(
        self, num_nodes: int, dof_types: list[DOFType] | None = None
    ) -> None:
        """Assign DOFs to all mesh nodes uniformly.

        Args:
            num_nodes: Total number of nodes in the mesh
            dof_types: List of DOF types to assign. If None, assigns all active types.

        Example:
            dof_space.assign_dofs_to_all_nodes(mesh.num_nodes)
        """
        for node in range(num_nodes):
            self.assign_dofs_to_node(node, dof_types)

    # -------------------------------------------------------------------------
    def get_global_dof(self, entity_key: EntityKey | int, dof_type: DOFType) -> int:
        """Get the global DOF index for a specific entity and DOF type.

        Args:
            entity_key: Tuple identifying the entity, or int for mesh node (auto-wrapped)
            dof_type: DOF type

        Returns:
            Global DOF index

        Raises:
            KeyError: If the entity or DOF type has not been assigned

        Example:
            global_dof = dof_space.get_global_dof((0,), DOFType.U_X)
            global_dof = dof_space.get_global_dof(0, DOFType.U_X)  # auto-wrapped
            global_dof = dof_space.get_global_dof(("fiber_1", 2), DOFType.U_Y)
        """
        # Auto-wrap integer node IDs as tuples
        if isinstance(entity_key, int):
            entity_key = (entity_key,)

        if entity_key not in self.dofs:
            raise KeyError(f"Entity {entity_key} has no DOFs assigned")
        if dof_type not in self.dofs[entity_key]:
            raise KeyError(
                f"Entity {entity_key} does not have DOF type {dof_type.value} assigned"
            )
        return self.dofs[entity_key][dof_type]

    # -------------------------------------------------------------------------
    @property
    def total_dofs(self) -> int:
        """Total number of DOFs in the system."""
        return self._next_global_dof
