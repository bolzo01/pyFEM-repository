#!/usr/bin/env python
"""
A Brown truss with a variable number of bays.

Created: 2025/10/31 18:35:08
Last modified: 2025/11/09 13:20:39
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def brown_truss(bays: int) -> tuple[list[list[int]], np.ndarray]:
    """Define the mesh (nodal coordinates and connectivity table) for a
    two-dimensional Brown truss with any number of bays."""

    bay = np.array([[0, 1], [0, 2]])
    bay_tmp = np.array([[1, 2], [1, 3], [2, 4], [1, 4], [2, 3]])
    bay = np.vstack((bay, bay_tmp))

    for _ in range(bays - 1):
        bay_tmp = bay_tmp + 2
        bay = np.vstack((bay, bay_tmp))

    bay_last = np.zeros((3, 2))
    bay_last[0, :] = bay_tmp[4, :] + 1
    bay_last[1, :] = (bay_tmp[4, 1], bay_tmp[4, 1] + 2)
    bay_last[2, :] = (bay_tmp[4, 1] + 1, bay_tmp[4, 1] + 2)

    connectivity = np.vstack((bay, bay_last)).astype(int)

    x_coords = np.vstack(
        (
            np.array([[0]]),
            np.repeat(np.arange(1, bays + 2), 2).reshape(-1, 1),
            np.array([[bays + 2]]),
        )
    )
    y_coords = np.vstack(
        (np.array([[0]]), np.tile([0, 1], bays + 1).reshape(-1, 1), np.array([[0]]))
    )
    points = np.hstack((x_coords, y_coords))

    return connectivity.tolist(), points


def main(bays: int = 3) -> float:
    # PREPROCESSING

    # 1. Geometry and discretization

    element_connectivity, points = brown_truss(bays)
    num_nodes = len(points)
    num_elements = len(element_connectivity)

    # 2. Element properties

    # Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            ("bar", ("bar_2D", {"E": 206000.0, "A": 500.0})),
        ]
    )

    # Assign properties to elements
    element_property_labels = ["bar"] * len(element_connectivity)

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
    model.bc.prescribe_displacement(num_nodes - 1, pyfem.DOFType.U_Y, 0.0)

    # Neumann boundary conditions (applied forces)
    model.bc.apply_force(1, pyfem.DOFType.U_Y, -10000.0)

    print(f"\n- Prescribed displacements: {model.bc.prescribed_displacements}")
    print(f"- Applied forces: {model.bc.applied_forces}")

    # PROCESSING: Solve FEA problem

    # Create solver
    solver = pyfem.LinearStaticSolver(model)

    # Assemble the global stiffness matrix
    solver.assemble_global_matrix()

    # Apply boundary conditions
    solver.apply_boundary_conditions()

    # Solve for nodal displacements
    nodal_displacements, original_global_stiffness_matrix = solver.solve()

    # POSTPROCESSING: Compute derived quantities

    # Create postprocessor
    postprocessor = pyfem.PostProcessor(
        model.mesh,
        model.element_properties,
        original_global_stiffness_matrix,
        nodal_displacements,
        magnification_factor=1000.0,
    )

    # Plot truss
    postprocessor.undeformed_mesh()
    postprocessor.deformed_mesh()

    return float(nodal_displacements[num_nodes - 1])


if __name__ == "__main__":
    main()
