#!/usr/bin/env python
"""
Solve a bar in tension.

Created: 2025/10/18 18:18:18
Last modified: 2025/10/19 02:56:53
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def main() -> None:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    bar_length = 12.0
    num_nodes = 2
    num_elements = 1
    dofs_per_node = 1

    # Nodal coordinates
    points = np.array([0.0, bar_length])

    # Element connectivity (which nodes belong to each element)
    element_connectivity = [
        [0, 1],
    ]

    # 2. Element properties

    # Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            ("bar", ("bar_1D", {"E": 23.2, "A": 7.0})),
        ]
    )

    # Assign properties to elements
    element_property_labels = ["bar"]

    # 3. Mesh

    # Create mesh
    mesh = pyfem.Mesh(
        num_nodes=num_nodes,
        points=points,
        num_elements=num_elements,
        element_connectivity=element_connectivity,
        element_property_labels=element_property_labels,
    )

    # Validate mesh and element properties
    pyfem.validate_mesh_and_element_properties(mesh, element_properties)

    # 4. Boundary conditions

    # Dirichlet boundary conditions (prescribed displacements)
    prescribed_displacements = [
        (0, 0.0),  # Node 0: fixed (u = 0)
    ]

    # Neumann boundary conditions (applied forces)
    applied_forces = [
        (1, 10.0),  # Node 1: force of 10.0
    ]

    # PROCESSING: Solve FEA problem

    # Create solver
    solver = pyfem.LinearStaticSolver(
        mesh,
        element_properties,
        applied_forces,
        prescribed_displacements,
        dofs_per_node,
    )

    # Assemble the global stiffness matrix
    solver.assemble_global_matrix()

    # Apply boundary conditions
    solver.apply_boundary_conditions()

    # Solve for nodal displacements
    nodal_displacements, original_global_stiffness_matrix = solver.solve()

    # POSTPROCESSING: Compute derived quantities

    # Create postprocessor
    postprocessor = pyfem.PostProcessor(
        mesh,
        element_properties,
        original_global_stiffness_matrix,
        nodal_displacements,
    )

    # Compute strain energy
    postprocessor.compute_strain_energy_global()


if __name__ == "__main__":
    main()
