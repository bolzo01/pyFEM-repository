#!/usr/bin/env python
"""
Solve a bar in tension.

Created: 2025/10/18 18:18:18
Last modified: 2025/11/15 12:22:45
Author: Francesco Bolzonella (francesco.bolzonella.1@studenti.unipd.it)
"""

import numpy as np

import pyfem


def main() -> np.ndarray:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    bar_length = 12.0
    num_nodes = 2
    num_elements = 1

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
            ("bar", ("bar_1D", {"E": 23.2, "A": 7.0}, {"integration": 3})),
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

    # Define node sets
    mesh.add_node_set(tag=1, nodes={0}, name="left_end")
    mesh.add_node_set(tag=2, nodes={1}, name="right_end")

    print("\n- Node sets:")
    for tag, node_set in mesh.node_sets.items():
        print(f"  {node_set}")

    # 4. Create Model

    problem = pyfem.Problem(
        pyfem.Physics.MECHANICS,
        pyfem.Dimension.D1,
    )

    model = pyfem.Model(mesh, problem)
    model.set_element_properties(element_properties)
    print(model)

    # 5. Boundary conditions

    # Dirichlet boundary conditions (prescribed displacements)
    model.bc.prescribe_displacement("left_end", pyfem.DOFType.U_X, 0.0)

    # Neumann boundary conditions (applied forces)
    model.bc.apply_force("right_end", pyfem.DOFType.U_X, 10.0)

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
    )

    # Compute strain energy
    postprocessor.compute_strain_energy_global()

    return solver.nodal_displacements


if __name__ == "__main__":
    main()
