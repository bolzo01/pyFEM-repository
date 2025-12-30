#!/usr/bin/env python
"""
Module defining the Mesh class for finite element analysis.

Created: 2024/10/13 19:05:39
Last modified: 2025/11/25 20:33:33
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import warnings

import meshio  # type: ignore
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

    @classmethod
    def from_gmsh(
        cls,
        filename: str,
        dim: int | None = None,
    ) -> "Mesh":
        """
        Build a Mesh from a Gmsh .msh file (via meshio).

        Supports:
            * dim = 2 -> 3-node triangles ("triangle") as domain elements
            * dim = 3 -> 4-node tets ("tetra") as domain elements

        Uses Gmsh physical names:
            * For domain elements: as element_property_labels (region names)
            * For boundary groups (lines in 2D, triangles in 3D): as node sets

        If a physical group has no name, an automatic name "region_<tag>" or
        "set_<tag>" is created and a warning is printed.

        Args:
            filename: Path to Gmsh .msh file
            dim: Problem dimension (2 or 3). If None, inferred from cell types.

        Returns:
            Mesh object with elements and node sets from Gmsh file

        Example:
            mesh = Mesh.from_gmsh("model.msh", dim=2)
            # Domain regions become element_property_labels
            # Boundary groups become node_sets
        """

        # Resolve relative path based on user's running script
        import os

        import __main__

        if not os.path.isabs(filename):
            try:
                base = os.path.dirname(os.path.abspath(__main__.__file__))
            except AttributeError:
                base = os.getcwd()  # interactive fallback
            filename = os.path.join(base, filename)

        gm = meshio.read(filename)

        # Decide spatial dimension and extract coordinates
        point_dim = gm.points.shape[1]  # usually 3 for Gmsh
        if dim is None:
            # Infer from physical entities: 2 -> triangles, 3 -> tets
            cell_types = set(ct for ct, _ in gm.cells)
            if "tetra" in cell_types:
                dim = 3
            elif "triangle" in cell_types:
                dim = 2
            else:
                raise ValueError(
                    "Cannot infer problem dimension: no 'triangle' or 'tetra' cells found."
                )
        else:
            if dim not in (2, 3):
                raise ValueError(f"Only dim=2 or dim=3 supported, got dim={dim}")

        # Trim points to the used dimension if needed
        if point_dim < dim:
            raise ValueError(
                f"Gmsh points have dimension {point_dim}, but dim={dim} was requested."
            )
        points = gm.points[:, :dim].astype(float)
        num_nodes = points.shape[0]

        # Build mapping physical_tag -> name, dimension
        # gm.field_data: name -> [tag, phys_dim]
        tag_to_name: dict[int, str] = {}
        tag_to_dim: dict[int, int] = {}

        for name, data in gm.field_data.items():
            # meshio often gives np.ndarray([tag, dim])
            tag = int(data[0])
            phys_dim = int(data[1])
            tag_to_name[tag] = str(name)
            tag_to_dim[tag] = phys_dim

        # Helper to get name or auto-name with warning
        def _region_name(tag: int, prefix: str) -> str:
            if tag in tag_to_name:
                return tag_to_name[tag]
            auto = f"{prefix}_{tag}"
            warnings.warn(
                f"Gmsh physical group with tag={tag} has no name; using '{auto}'.",
                UserWarning,
            )
            return auto

        # Convenience: access cell blocks as dicts
        # meshio offers cells_dict and cell_data_dict
        try:
            cells_dict = gm.cells_dict  # type: ignore[attr-defined]
            cell_data_dict = gm.cell_data_dict  # type: ignore[attr-defined]
        except AttributeError:
            # Fallback for older meshio: construct dicts manually
            cells_dict = {}  # <-- NO type annotation here
            for cell_block in gm.cells:
                cell_type = cell_block.type
                data = cell_block.data
                cells_dict[cell_type] = data

            cell_data_dict = {}  # <-- NO type annotation here
            for name, data_list in gm.cell_data.items():
                cell_data_dict[name] = {}
                for cell_block, arr in zip(gm.cells, data_list):
                    cell_data_dict[name][cell_block.type] = arr

        # Physical tags per cell type
        if "gmsh:physical" in cell_data_dict:
            phys_by_type: dict[str, np.ndarray] = cell_data_dict["gmsh:physical"]
        else:
            raise ValueError(
                "Gmsh mesh has no 'gmsh:physical' cell_data. "
                "Enable physical groups in your .geo/.msh export."
            )

        # Extract domain elements (triangle/tetra) + region names
        element_connectivity: list[list[int]] = []
        element_property_labels: list[str] = []

        # Domain cell types we care about
        if dim == 2:
            domain_type = "triangle"
            required_phys_dim = 2
        else:  # dim == 3
            domain_type = "tetra"
            required_phys_dim = 3

        if domain_type not in cells_dict:
            raise ValueError(
                f"No '{domain_type}' cells found for dim={dim} in Gmsh file '{filename}'."
            )
        if domain_type not in phys_by_type:
            raise ValueError(
                f"No gmsh:physical data for '{domain_type}' cells. "
                "Check your Gmsh physical group definitions."
            )

        domain_cells = cells_dict[domain_type]  # (n_elems, n_nodes)
        domain_phys_tags = phys_by_type[domain_type]  # (n_elems,)

        if len(domain_cells) != len(domain_phys_tags):
            raise ValueError(
                f"Mismatch between number of '{domain_type}' cells "
                f"({len(domain_cells)}) and physical tags ({len(domain_phys_tags)})."
            )

        for conn, tag_val in zip(domain_cells, domain_phys_tags):
            tag = int(tag_val)
            # Optional: cross-check physical dimension (from field_data)
            phys_dim = tag_to_dim.get(tag, required_phys_dim)
            if phys_dim != required_phys_dim:
                warnings.warn(
                    f"Domain element with physical tag={tag} has phys_dim={phys_dim}, "
                    f"but problem dim={dim}. Including it anyway.",
                    UserWarning,
                )

            region = _region_name(tag, "region")
            element_connectivity.append(conn.tolist())
            element_property_labels.append(region)

        num_elements = len(element_connectivity)

        # Create Mesh object
        mesh = cls(
            num_nodes=num_nodes,
            points=points,
            num_elements=num_elements,
            element_connectivity=element_connectivity,
            element_property_labels=element_property_labels,
        )

        # Build node sets from lower-dimension physical groups (boundaries for BCs)
        #   - dim=2 -> 1D "line" elements
        #   - dim=3 -> 2D "triangle" elements
        node_sets_nodes: dict[int, set[int]] = {}

        if dim == 2:
            boundary_type = "line"
            boundary_phys_dim = 1
        else:  # dim == 3
            boundary_type = "triangle"
            boundary_phys_dim = 2

        if boundary_type in cells_dict and boundary_type in phys_by_type:
            b_cells = cells_dict[boundary_type]
            b_tags = phys_by_type[boundary_type]

            if len(b_cells) != len(b_tags):
                warnings.warn(
                    f"Mismatch between number of '{boundary_type}' boundary cells "
                    f"({len(b_cells)}) and physical tags ({len(b_tags)}). "
                    "Boundary node sets may be incomplete.",
                    UserWarning,
                )

            for conn, tag_val in zip(b_cells, b_tags):
                tag = int(tag_val)
                # Optional dimension check
                phys_dim = tag_to_dim.get(tag, boundary_phys_dim)
                if phys_dim != boundary_phys_dim:
                    # This physical group is not truly a boundary of desired dim
                    continue

                if tag not in node_sets_nodes:
                    node_sets_nodes[tag] = set()
                node_sets_nodes[tag].update(int(i) for i in conn)

        # Add physical point node sets (dimension 0)
        if "vertex" in cells_dict and "vertex" in phys_by_type:
            v_cells = cells_dict["vertex"]
            v_tags = phys_by_type["vertex"]
            for conn, tag_val in zip(v_cells, v_tags):
                tag = int(tag_val)
                if tag not in node_sets_nodes:
                    node_sets_nodes[tag] = set()
                node_sets_nodes[tag].add(int(conn[0]))

        # Register node sets in the Mesh
        for tag, nodes in node_sets_nodes.items():
            name = _region_name(tag, "set")
            mesh.add_node_set(tag=tag, nodes=nodes, name=name)

        return mesh
