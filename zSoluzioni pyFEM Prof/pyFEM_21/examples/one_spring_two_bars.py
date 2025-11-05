#!/usr/bin/env python
"""
Solve a series combination of one spring and two bars in tension.

Created: 2025/10/18 22:16:45
Last modified: 2025/10/27 09:39:51
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def main() -> np.ndarray:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    L = 1
    num_nodes = 4
    num_elements = 3

    # Nodal coordinates
    points = np.array([0.0, L, 2 * L, 3 * L])

    # Element connectivity (which nodes belong to each element)
    element_connectivity = [
        [0, 1],
        [1, 2],
        [2, 3],
    ]

    # 2. Element properties

    # Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            ("bar1", ("bar_1D", {"E": 2.0, "A": 2.0})),
            ("bar2", ("bar_1D", {"E": 2.0, "A": 1.0})),
            ("spring", ("spring_1D", {"k": 2.0})),
        ]
    )

    # Assign properties to elements
    element_property_labels = ["spring", "bar1", "bar2"]

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

    # 4. DOF Space Setup

    # Create DOF space and activate the displacement DOF
    dof_space = pyfem.DOFSpace()
    dof_space.activate_dof_types(pyfem.DOFType.U_X)

    # Assign DOFs to all nodes (each node gets one DOF: U_X)
    dof_space.assign_dofs_to_all_nodes(mesh.num_nodes)

    print(f"\n- DOF Space: {dof_space}")

    # 5. Boundary conditions

    # Dirichlet boundary conditions (prescribed displacements)
    prescribed_displacements = [
        (dof_space.get_global_dof(0, pyfem.DOFType.U_X), 0.0),
        (dof_space.get_global_dof(3, pyfem.DOFType.U_X), 4.0),
    ]

    # Neumann boundary conditions (applied forces)
    applied_forces = None

    # PROCESSING: Solve FEA problem

    # Create solver
    solver = pyfem.LinearStaticSolver(
        mesh,
        element_properties,
        applied_forces,
        prescribed_displacements,
        dof_space,
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

    return nodal_displacements


if __name__ == "__main__":
    main()
