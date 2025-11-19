#!/usr/bin/env python
"""
Solve a series combination of two 1D springs.

Created: 2025/08/02 17:32:52
Last modified: 2025/11/16 02:40:48
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def main() -> np.ndarray:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    num_nodes = 3
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

    # Define node sets
    mesh.add_node_set(tag=1, nodes={1}, name="left_end")
    mesh.add_node_set(tag=2, nodes={0}, name="right_end")

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
    solution = solver.solve()

    # POSTPROCESSING: Analyze results

    # Create postprocessor
    postprocessor = pyfem.PostProcessor(
        model=model,
        solution=solution,
        global_stiffness_matrix=solver.global_stiffness_matrix,
    )

    # - Compute strain energy at element and system levels
    postprocessor.compute_strain_energy_local()
    postprocessor.compute_strain_energy_global()

    return solution.nodal_displacements


if __name__ == "__main__":
    main()
