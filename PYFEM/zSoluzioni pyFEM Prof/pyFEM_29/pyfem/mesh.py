#!/usr/bin/env python
"""
Module defining the Mesh class for finite element analysis.

Created: 2024/10/13 19:05:39
Last modified: 2025/10/27 22:38:40
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from .node_set import NodeSet


class Mesh:
    """Represents the finite element mesh data structure.

    Stores discretization information including nodal coordinates, element
    connectivity, element property assignments, and node sets.

    Attributes:
        num_nodes: Number of nodes in the mesh
        points: Nodal coordinates
        num_elements: Number of elements in the mesh
        element_connectivity: Element connectivity matrix
        element_property_labels: Per-element property labels
        node_sets: Dictionary of node sets indexed by tag (integer ID)
        node_set_names: Optional mapping from names to tags for convenience
    """

    def __init__(
        self,
        num_nodes: int,
        points: np.ndarray,
        num_elements: int,
        element_connectivity: list[list[int]],
        element_property_labels: list[str],
    ):
        # Geometry and topology
        self.num_nodes = num_nodes
        self.points = points
        self.num_elements = num_elements
        self.element_connectivity = element_connectivity

        # Element properties
        self.element_property_labels = element_property_labels

        # Node sets (for boundary conditions, material regions, etc.)
        self.node_sets: dict[int, NodeSet] = {}  # {tag: NodeSet}
        self.node_set_names: dict[str, int] = {}  # {name: tag} - optional lookup

    def add_node_set(self, tag: int, nodes: set[int], name: str | None = None) -> None:
        """Add a node set to the mesh.

        Args:
            tag: Integer identifier (matches Gmsh physical group ID)
            nodes: Set of node IDs
            name: Optional name for the node set

        Example:
            mesh.add_node_set(tag=1, nodes={0, 1}, name="left_boundary")
        """
        if tag in self.node_sets:
            raise ValueError(f"Node set with tag {tag} already exists")

        node_set = NodeSet(tag=tag, nodes=nodes, name=name)
        self.node_sets[tag] = node_set

        # Add to name lookup if name provided
        if name is not None:
            if name in self.node_set_names:
                raise ValueError(f"Node set with name '{name}' already exists")
            self.node_set_names[name] = tag

    def get_node_set(self, tag_or_name: int | str) -> NodeSet:
        """Get a node set by tag (int) or name (str).

        Example:
            node_set = mesh.get_node_set(1)  # by tag
            node_set = mesh.get_node_set("left_boundary")  # by name
        """
        if isinstance(tag_or_name, int):
            if tag_or_name not in self.node_sets:
                raise KeyError(f"Node set with tag {tag_or_name} not found")
            return self.node_sets[tag_or_name]
        elif isinstance(tag_or_name, str):
            if tag_or_name not in self.node_set_names:
                raise KeyError(f"Node set with name '{tag_or_name}' not found")
            tag = self.node_set_names[tag_or_name]
            return self.node_sets[tag]
        else:
            raise TypeError(f"tag_or_name must be int or str, got {type(tag_or_name)}")
