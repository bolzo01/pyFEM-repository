#!/usr/bin/env python
"""
Two 3-node bars in series with varying cross sections, under tension.

Created: 2025/11/11 18:18:18
Last modified: 2025/11/17 23:50:48
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def main() -> np.ndarray:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    domain_size = 2
    num_nodes = 5
    num_elements = 2

    # Nodal coordinates
    points = np.linspace(0.0, domain_size, num=num_nodes, endpoint=True)

    # Element connectivity (which nodes belong to each element)
    element_connectivity = [
        [0, 2, 1],
        [2, 4, 3],
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
                "thick_bar",
                pyfem.ElementProperty(
                    "bar3_1D", {"A": 2.0}, material="mate1", meta={"integration": 5}
                ),
            ),
            (
                "thin_bar",
                pyfem.ElementProperty(
                    "bar3_1D", {"A": 1.0}, material="mate1", meta={"integration": 2}
                ),
            ),
        ]
    )

    # Assign properties to elements
    element_property_labels = ["thick_bar", "thin_bar"]

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
    mesh.add_node_set(tag=2, nodes={4}, name="right_end")

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

    # Neumann boundary conditions (applied forces)
    model.bc.apply_force("right_end", pyfem.DOFType.U_X, 4.0)

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

    # Compute strain energy
    postprocessor.compute_strain_energy_global()

    postprocessor.compute_element_stresses()

    print("stress", solution.element_stresses)

    return solution.nodal_displacements


if __name__ == "__main__":
    main()
