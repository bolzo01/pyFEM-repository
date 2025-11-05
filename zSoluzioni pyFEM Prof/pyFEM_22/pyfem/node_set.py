#!/usr/bin/env python
"""
Module defining NodeSet class for grouping mesh nodes.

Created: 2025/10/25 19:26:26
Last modified: 2025/10/26 00:03:01
Author: Angelo Simone (angelo.simone@unipd.it)
"""


class NodeSet:
    """Container for a group of nodes identified by geometric or topological criteria.

    NodeSets are used to:
    - Apply boundary conditions to groups of nodes
    - Define material regions or domains
    - Correspond to Gmsh Physical Groups

    Attributes:
        tag: Integer identifier (matches Gmsh physical group ID)
        name: Optional human-readable name
        nodes: Set of node IDs belonging to this group
    """

    def __init__(self, tag: int, nodes: set[int], name: str | None = None):
        """Initialize a NodeSet.

        Args:
            tag: Integer identifier (primary key, matches Gmsh physical tag)
            nodes: Set of node IDs
            name: Optional name for the node set

        Example:
            node_set = NodeSet(tag=1, nodes={0, 1, 2}, name="left_boundary")
        """
        self.tag = tag
        self.nodes = nodes
        self.name = name

    def __len__(self) -> int:
        """Return the number of nodes in this set."""
        return len(self.nodes)

    def __iter__(self):
        """Allow iteration over nodes in the set."""
        return iter(self.nodes)

    def __contains__(self, node: int) -> bool:
        """Check if a node belongs to this set."""
        return node in self.nodes

    def __repr__(self) -> str:
        name_str = f", name='{self.name}'" if self.name else ""
        return f"NodeSet(tag={self.tag}, num_nodes={len(self.nodes)}{name_str})"
