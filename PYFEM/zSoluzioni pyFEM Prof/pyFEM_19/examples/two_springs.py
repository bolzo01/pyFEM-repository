#!/usr/bin/env python
"""
Solve a series combination of two 1D springs.

Created: 2025/08/02 17:32:52
Last modified: 2025/10/19 17:44:36
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def main() -> np.ndarray:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    num_nodes = 3
    dofs_per_node = 1
    num_elements = 2

    # Nodal coordinates
    points = np.array([0.0, 1.0, 2.0, 3.0])

    # Element connectivity (which nodes belong to each element)
    element_connectivity = [
        [1, 2],
        [2, 0],
    ]

    # 2. Element properties

    # Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            ("soft", ("spring_1D", {"k": 1.0})),
            ("stiff", ("spring_1D", {"k": 2.0})),
        ]
    )

    # Assign properties to elements
    element_property_labels = ["soft", "stiff"]

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
        (1, 0.0),  # DOF 0 is constrained
    ]

    # Neumann boundary conditions (applied forces)
    applied_forces = [
        (0, 10.0),  # DOF 1 has an applied force of 10
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

    # - Compute strain energy at element and system levels
    postprocessor.compute_strain_energy_local()
    postprocessor.compute_strain_energy_global()

    return nodal_displacements


if __name__ == "__main__":
    main()
