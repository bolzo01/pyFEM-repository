#!/usr/bin/env python
"""
Module for FEA procedures.

Created: 2025/10/08 17:11:28
Last modified: 2025/11/17 22:06:55
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np
from scipy import sparse

from .dof_types import DOFSpace
from .element_properties import ElementProperties
from .elements import create_element
from .materials import Material
from .mesh import Mesh


def assemble_global_stiffness_matrix(
    mesh: Mesh,
    element_properties: ElementProperties,
    materials: dict[str, Material],
    global_stiffness_matrix: np.ndarray,
    dof_space: DOFSpace,
    use_sparse: bool = True,
) -> np.ndarray | sparse.spmatrix:
    """
    Assembles the global stiffness matrix by integrating element stiffness matrices.

    Returns:
        The assembled global stiffness matrix (dense or sparse).
    """

    num_elements = mesh.num_elements
    element_connectivity = mesh.element_connectivity

    # ============================================================
    # SPARSE ASSEMBLY (with duplicate elimination)
    # ============================================================
    if use_sparse:
        # Use dictionary to accumulate values and eliminate duplicates
        triplet_dict: dict[tuple[int, int], float] = {}

        for element_index in range(num_elements):
            # Retrieve the element property definition (kind, params, meta)
            label = mesh.element_property_labels[element_index]
            elem_prop = element_properties[label]

            # Instantiate the element object through the element registry
            elem = create_element(elem_prop)

            # Get the indices of the mesh nodes that belong to this element
            element_nodes = element_connectivity[element_index]

            # Extract the physical coordinates of those nodes
            x_nodes = mesh.points[element_nodes]

            # Resolve the material associated with an element property.
            material = resolve_material(label, elem_prop, materials)

            # Compute the element stiffness matrix using the element formulation
            local_K = elem.compute_stiffness(material=material, x_nodes=x_nodes)

            # Get DOF mapping
            dof_map = dof_space.get_dof_mapping(element_connectivity[element_index])

            # Accumulate into dictionary (sums duplicates manually)
            for i, i_global in enumerate(dof_map):
                for j, j_global in enumerate(dof_map):
                    key = (i_global, j_global)
                    triplet_dict[key] = triplet_dict.get(key, 0.0) + local_K[i, j]

        # Remove entries that cancel to ~0 due to numerical roundoff
        tolerance = 1e-13
        filtered_triplets = [
            (i, j, v) for (i, j), v in triplet_dict.items() if abs(v) > tolerance
        ]

        n_entries = len(filtered_triplets)
        data = np.empty(n_entries, dtype=float)
        row = np.empty(n_entries, dtype=int)
        col = np.empty(n_entries, dtype=int)

        for idx, (i, j, value) in enumerate(filtered_triplets):
            row[idx] = i
            col[idx] = j
            data[idx] = value

        # Single conversion: COO -> CSC
        n_dof = dof_space.total_dofs
        K_csc = sparse.csc_matrix((data, (row, col)), shape=(n_dof, n_dof))

        return K_csc

    # ============================================================
    # DENSE ASSEMBLY
    # ============================================================
    else:
        # Assemble the global stiffness matrix
        # print("\n- Assembling local stiffness matrix into global stiffness matrix")
        for element_index in range(num_elements):
            # Retrieve the element property definition (kind, params, meta)
            label = mesh.element_property_labels[element_index]
            elem_prop = element_properties[label]

            # Instantiate the element object through the element registry
            elem = create_element(elem_prop)

            # Get the indices of the mesh nodes that belong to this element
            element_nodes = element_connectivity[element_index]

            # Extract the physical coordinates of those nodes
            x_nodes = mesh.points[element_nodes]

            # Resolve the material associated with an element property.
            material = resolve_material(label, elem_prop, materials)

            # Compute the element stiffness matrix using the element formulation
            local_K = elem.compute_stiffness(material=material, x_nodes=x_nodes)

            # Get local to global DOF mapping using DOFSpace
            dof_mapping = dof_space.get_dof_mapping(element_nodes)

            # Assemble the local stiffness matrix into the global stiffness matrix
            for i, global_i in enumerate(dof_mapping):
                for j, global_j in enumerate(dof_mapping):
                    global_stiffness_matrix[global_i, global_j] += local_K[i, j]

        return global_stiffness_matrix


def resolve_material(
    label: str, elem_prop, materials: dict[str, Material]
) -> Material | None:
    """
    Resolve the material for an element.
    Returns a Material instance or None for material-free elements (e.g. springs).
    """

    material_name = elem_prop.material

    # If no material is specified, return None (allowed for some elements)
    if material_name is None:
        return None

    # Material must exist
    if material_name not in materials:
        raise ValueError(
            f"Material '{material_name}' for element '{label}' not found in model.materials."
        )

    return materials[material_name]
