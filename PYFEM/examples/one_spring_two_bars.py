#!/usr/bin/env python
"""
Solve a series combination of one spring and two bars in tension.

Created: 2025/10/18 22:16:45
Last modified: 2025/11/17 23:54:23
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

    # 2. Materials
    materials = pyfem.make_materials(
        [
            ("mate1", pyfem.LinearElastic1D(E=2.0)),
        ]
    )

    # 3. Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            (
                "bar1",
                pyfem.ElementProperty("bar_1D", {"A": 2.0}, material="mate1"),
            ),
            (
                "bar2",
                pyfem.ElementProperty("bar_1D", {"A": 1.0}, material="mate1"),
            ),
            (
                "spring",
                pyfem.ElementProperty(kind="spring_1D", params={"k": 2.0}),
            ),
        ]
    )

    # Assign properties to elements
    element_property_labels = ["spring", "bar1", "bar2"]

    # 4. Mesh

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
    mesh.add_node_set(tag=2, nodes={3}, name="right_end")

    print("\n- Node sets:")
    for tag, node_set in mesh.node_sets.items():
        print(f"  {node_set}")

    # 5. Create Model

    problem = pyfem.Problem(
        pyfem.Physics.MECHANICS,
        pyfem.Dimension.D1,
    )

    model = pyfem.Model(mesh, problem)
    model.set_materials(materials)
    model.set_element_properties(element_properties)
    print(model)

    # 6. Boundary conditions

    # Dirichlet boundary conditions (prescribed displacements)
    model.bc.prescribe_displacement("left_end", pyfem.DOFType.U_X, 0.0)
    model.bc.prescribe_displacement("right_end", pyfem.DOFType.U_X, 4.0)

    # print(f"\n- Prescribed displacements: {model.bc.prescribed_displacements}")
    # print(f"- Applied forces: {model.bc.applied_forces}")

    # PROCESSING: Solve FEA problem

    # Create solver from model
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

    # Compute strain energy
    postprocessor.compute_strain_energy_global()

    postprocessor.compute_element_stresses()

    print("stress", solution.element_stresses)

    return solution.nodal_displacements


if __name__ == "__main__":
    main()
