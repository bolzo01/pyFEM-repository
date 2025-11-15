#!/usr/bin/env python
"""
Demonstrates detection of conflicting boundary conditions.

Created: 2024/11/06 17:32:52
Last modified: 2025/11/08 17:03:36
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def main() -> np.ndarray:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    L1 = 4000.0
    L2 = 6000.0
    q1 = -10 * L1
    num_nodes = 9
    num_elements = 15

    # Nodal coordinates
    points = np.array(
        [
            [0, 0],
            [L1, 0],
            [2 * L1, 0],
            [3 * L1, 0],
            [4 * L1, 0],
            [0.5 * L1, L2],
            [1.5 * L1, L2],
            [2.5 * L1, L2],
            [3.5 * L1, L2],
        ]
    )

    # Element connectivity (which nodes belong to each element)
    element_connectivity = [
        [0, 5],
        [0, 1],
        [5, 1],
        [5, 6],
        [1, 6],
        [1, 2],
        [6, 2],
        [6, 7],
        [2, 7],
        [2, 3],
        [7, 3],
        [7, 8],
        [3, 8],
        [3, 4],
        [8, 4],
    ]

    # 2. Element properties

    # Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            ("bar", ("bar_2D", {"E": 206000.0, "A": 500.0})),
        ]
    )

    # Assign properties to elements
    element_property_labels = ["bar"] * num_elements

    # 3. Mesh

    # Create mesh
    mesh = pyfem.Mesh(
        num_nodes=num_nodes,
        points=points,
        num_elements=num_elements,
        element_connectivity=element_connectivity,
        element_property_labels=element_property_labels,
    )

    # 4. Create Model

    problem = pyfem.Problem(
        pyfem.Physics.MECHANICS,
        pyfem.Dimension.D2,
    )

    model = pyfem.Model(mesh, problem)
    model.set_element_properties(element_properties)
    print(model)

    # 5. Boundary conditions

    # Dirichlet boundary conditions (prescribed displacements)
    model.bc.prescribe_displacement(0, pyfem.DOFType.U_X, 0.0)
    model.bc.prescribe_displacement(0, pyfem.DOFType.U_Y, 0.0)
    model.bc.prescribe_displacement(4, pyfem.DOFType.U_Y, 0.0)

    # Neumann boundary conditions (applied forces)
    model.bc.apply_force(0, pyfem.DOFType.U_Y, q1)  # <-- already assigned Dirichlet
    model.bc.apply_force(1, pyfem.DOFType.U_Y, 2.0 * q1)
    model.bc.apply_force(2, pyfem.DOFType.U_Y, 2.0 * q1)
    model.bc.apply_force(3, pyfem.DOFType.U_Y, 2.0 * q1)
    model.bc.apply_force(4, pyfem.DOFType.U_Y, q1)  #  <-- already assigned Dirichlet

    # print(f"\n- Prescribed displacements: {model.bc.prescribed_displacements}")
    # print(f"- Applied forces: {model.bc.applied_forces}")

    # PROCESSING: Solve FEA problem

    # Create solver
    solver = pyfem.LinearStaticSolver(model)

    # Assemble the global stiffness matrix
    solver.assemble_global_matrix()

    # Apply boundary conditions
    solver.apply_boundary_conditions()

    # Solve for nodal displacements
    solver.solve()

    # POSTPROCESSING: Compute derived quantities

    # Create postprocessor
    postprocessor = pyfem.PostProcessor(
        model.mesh,
        model.element_properties,
        solver.global_stiffness_matrix,
        solver.nodal_displacements,
        magnification_factor=100.0,
    )

    # Compute strain energy
    postprocessor.compute_strain_energy_global()

    # Plot truss
    postprocessor.undeformed_mesh()
    postprocessor.deformed_mesh()

    print("Solution\n", solver.nodal_displacements)

    return solver.nodal_displacements


if __name__ == "__main__":
    main()
