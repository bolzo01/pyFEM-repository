#!/usr/bin/env python
"""
Solve a series combination of two 1D springs.

Created: 2025/08/02 17:32:52
Last modified: 2025/10/29 11:05:28
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import pyfem


def main() -> None:
    # Preprocessing

    # - Define input data
    num_nodes = 3
    dofs_per_node = 1
    num_elements = 2

    # - Define discretization

    # -- Connectivity matrix defining which nodes belong to each element
    element_connectivity = [
        [1, 2],
        [2, 0],
    ]

    # - Define material properties
    # -- Stiffness properties for each spring element
    # Materials registry as list of (label, entry) pairs
    materials = pyfem.make_materials(
        [
            (
                "soft",
                ("spring_1D", {"k": 1.0}),
            ),
            (
                "stiff",
                ("spring_1D", {"k": 2.0}),
            ),
        ]
    )

    # Per-element material labels
    element_material = ["soft", "stiff"]

    # Mesh object
    mesh = pyfem.Mesh(
        num_nodes=num_nodes,
        num_elements=num_elements,
        element_connectivity=element_connectivity,
        element_material=element_material,
    )

    # Validate mesh and materials
    pyfem.validate_mesh_and_materials(mesh, materials)

    # - Define boundary conditions

    # -- Prescribed displacements (Dirichlet boundary conditions): [DOF, value]
    prescribed_displacements = [
        (1, 0.0),  # DOF 0 is constrained
    ]

    # -- Applied forces (Neumann boundary conditions): [DOF, value]
    applied_forces = [
        (0, 10.0),  # DOF 1 has an applied force of 10
    ]

    # Processing: Calculate the nodal displacements

    # - Instantiate the solver class
    solver = pyfem.LinearStaticSolver(
        mesh,
        materials,
        applied_forces,
        prescribed_displacements,
        dofs_per_node,
    )

    # - Assemble the global stiffness matrix
    solver.assemble_global_matrix()

    # - Apply boundary conditions
    solver.apply_boundary_conditions()

    # - Solve KU=F for the displacement vector
    nodal_displacements, original_global_stiffness_matrix = solver.solve()

    # Postprocessing: Calculate strain energy for each spring and for system of springs

    # - Compute strain energy at element level
    pyfem.compute_strain_energy_local(mesh, materials, nodal_displacements)

    # - Compute strain energy at system level
    pyfem.compute_strain_energy_global(
        original_global_stiffness_matrix, nodal_displacements
    )


if __name__ == "__main__":
    main()
