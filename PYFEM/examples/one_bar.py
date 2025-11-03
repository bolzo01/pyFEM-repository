#!/usr/bin/env python
"""
Solve a series combination of two 1D springs.

Created: 2025/08/02 17:32:52
Last modified: 2025/11/03 11:57:57
Author: Francesco Bolzonella (francesco.bolzonella.1@studentiunipd.it)
"""

import numpy as np

import pyfem


def main() -> None:
    # Preprocessing

    # - Define input data
    num_nodes = 2
    dofs_per_node = 1
    num_elements = 1

    # - Define discretization

    # -- Bar lenght
    bar_length = 12

    # -- Nodal coordinates
    points = np.array([0.0, bar_length])

    # -- Connectivity matrix defining which nodes belong to each element
    element_connectivity = [
        [0, 1],
    ]

    # - Define material properties
    # -- Stiffness properties for each spring element
    # Materials registry as list of (label, entry) pairs
    materials = pyfem.make_materials(
        [
            (
                "bar",
                ("bar_1D", {"E": 23.2, "A": 7.0}),
            )
        ]
    )

    # Per-element material labels
    element_material = ["bar"]

    # Mesh object
    mesh = pyfem.Mesh(
        num_nodes=num_nodes,
        points=points,
        num_elements=num_elements,
        element_connectivity=element_connectivity,
        element_material=element_material,
    )

    # Validate mesh and materials
    pyfem.validate_mesh_and_element_properties(mesh, materials)

    # - Define boundary conditions

    # -- Prescribed displacements (Dirichlet boundary conditions): [DOF, value]
    prescribed_displacements = [
        (0, 0.0),  # DOF 0 is constrained
    ]

    # -- Applied forces (Neumann boundary conditions): [DOF, value]
    applied_forces = [
        (1, 10.0),  # DOF 1 has an applied force of 10
    ]

    # Processing: Calculate the nodal displacements

    # Instantiate the solver class
    solver = pyfem.LinearStaticSolver(
        mesh,
        materials,
        applied_forces,
        prescribed_displacements,
        dofs_per_node,
    )

    # Assemble the global stiffness matrix
    solver.assemble_global_matrix()

    # Apply boundary conditions
    solver.apply_boundary_conditions()

    # Solve KU=F for the displacement vector
    nodal_displacements, original_global_stiffness_matrix = solver.solve()

    # Postprocessing: Calculate strain energy for each spring and for system of springs

    # - Instantiate the postprocessor class
    postprocessor = pyfem.PostProcessor(
        mesh,
        materials,
        original_global_stiffness_matrix,
        nodal_displacements,
    )

    # - Compute strain energy at element and system levels
    postprocessor.compute_strain_energy_local()
    postprocessor.compute_strain_energy_global()


if __name__ == "__main__":
    main()
