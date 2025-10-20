#!/usr/bin/env python
"""
Solve a series combination of two 1D springs.

Created: 2025/08/02 17:32:52
Last modified: 2025/10/13 11:08:55
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

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

    # - Initialize arrays

    # -- Compute total number of DOFs
    total_dofs = dofs_per_node * mesh.num_nodes

    # -- Initialize the global stiffness matrix as a square matrix of zeros
    global_stiffness_matrix = np.zeros((total_dofs, total_dofs))

    # -- Initialize the global force vector with zeros
    global_force_vector = np.zeros(total_dofs)

    # Processing

    # - Assemble the global stiffness matrix
    pyfem.assemble_global_stiffness_matrix(mesh, materials, global_stiffness_matrix)
    print("\n- Global stiffness matrix K:")
    for row in global_stiffness_matrix:
        print(row)

    # - Save a copy of the original global stiffness matrix before applying boundary conditions
    original_global_stiffness_matrix = global_stiffness_matrix.copy()

    # - Boundary conditions: Apply forces
    pyfem.apply_nodal_forces(applied_forces, global_force_vector)

    # - Boundary conditions: Constrain displacements
    pyfem.apply_prescribed_displacements(
        prescribed_displacements,
        global_stiffness_matrix,
        global_force_vector,
        total_dofs,
    )

    print("\n- Modified global stiffness matrix K after applying boundary conditions:")
    for row in global_stiffness_matrix:
        print(row)

    print("\n- Global force vector F after applying boundary conditions:")
    print(global_force_vector)

    # - Solve for the nodal displacements
    nodal_displacements = np.linalg.solve(global_stiffness_matrix, global_force_vector)
    print("\n- Nodal displacements U:")
    print(nodal_displacements)

    # Postprocessing: Calculate strain energy for each spring and for system of springs

    # - Compute strain energy at element level
    pyfem.compute_strain_energy_local(mesh, materials, nodal_displacements)

    # - Compute strain energy at system level
    pyfem.compute_strain_energy_global(
        original_global_stiffness_matrix, nodal_displacements
    )


if __name__ == "__main__":
    main()
