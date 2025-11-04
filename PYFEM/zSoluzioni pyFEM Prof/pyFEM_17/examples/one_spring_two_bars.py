#!/usr/bin/env python
"""
Solve a series combination of one spring and two bars in tension.

Created: 2025/10/18 22:16:45
Last modified: 2025/10/18 22:40:13
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def main() -> None:
    # Preprocessing

    # - Define input data
    num_nodes = 4
    dofs_per_node = 1
    num_elements = 3

    # - Define discretization

    # -- Nodal coordinates
    L = 1
    points = np.array([0.0, L, 2 * L, 3 * L])

    # -- Connectivity matrix defining which nodes belong to each element
    element_connectivity = [
        [0, 1],
        [1, 2],
        [2, 3],
    ]

    # - Define material properties
    # -- Stiffness properties for each spring element
    # Materials registry as list of (label, entry) pairs
    materials = pyfem.make_materials(
        [
            ("bar1", ("bar_1D", {"E": 2.0, "A": 2.0})),
            ("bar2", ("bar_1D", {"E": 2.0, "A": 1.0})),
            ("spring", ("spring_1D", {"k": 2.0})),
        ]
    )

    # Per-element material labels
    element_material = ["spring", "bar1", "bar2"]

    # Mesh object
    mesh = pyfem.Mesh(
        num_nodes=num_nodes,
        points=points,
        num_elements=num_elements,
        element_connectivity=element_connectivity,
        element_material=element_material,
    )

    # Validate mesh and materials
    pyfem.validate_mesh_and_materials(mesh, materials)

    # - Define boundary conditions

    # -- Prescribed displacements (Dirichlet boundary conditions): [DOF, value]
    prescribed_displacements = [
        (0, 0.0),
    ]

    # -- Applied forces (Neumann boundary conditions): [DOF, value]
    applied_forces = [
        (3, 4.0),
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

    # - Instantiate the postprocessor class
    postprocessor = pyfem.PostProcessor(
        mesh,
        materials,
        original_global_stiffness_matrix,
        nodal_displacements,
    )

    # - Compute strain energy
    postprocessor.compute_strain_energy_global()


if __name__ == "__main__":
    main()
