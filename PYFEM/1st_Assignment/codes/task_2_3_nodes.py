#!/usr/bin/env python

import numpy as np

import pyfem


def main() -> np.ndarray:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    bar_length = 10.0
    num_nodes = 3
    number_elements = np.array([1.0])
    P = -1.0
    E = 1.0
    A = 1.0
    # Parameters required by the postprocessor
    alpha_values, alpha_counter = np.array([1.0]), 0
    alpha = alpha_values[alpha_counter]
    epsilon_h = np.zeros((len(alpha_values), len(num_elements)))
    i = np.where(number_elements == num_elements)[0][0]

    # Nodal coordinates
    points = np.array(
        [
            0.0,
            bar_length / 2.0,
            bar_length,
        ]
    )

    # Element connectivity (which nodes belong to each element)
    element_connectivity = [
        [0, 1, 2],
    ]

    # 2. Element properties

    # Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            ("bar", ("bar3_1D", {"E": 1.0, "A": 1.0, "k": 1.0})),
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
    mesh.add_node_set(tag=2, nodes={2}, name="right_end")

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
    model.bc.prescribe_displacement(
        "right_end", pyfem.DOFType.U_X, (-(P / (E * A * alpha)) * np.exp(-alpha * 10.0))
    )

    # Neumann boundary conditions (applied forces)
    model.bc.apply_force("left_end", pyfem.DOFType.U_X, P)

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
    solver.solve()

    # POSTPROCESSING: Compute derived quantities

    number_elements = np.array([1.0])

    # Create postprocessor
    postprocessor = pyfem.PostProcessor(
        model.mesh,
        model.element_properties,
        solver.global_stiffness_matrix,
        solver.nodal_displacements,
        number_elements,
        alpha,
        alpha_counter,
        alpha_values,
        E,
        A,
        P,
    )

    # Compute strain energy
    epsilon_h = postprocessor.compute_strain_energy_global(i, epsilon_h)

    return nodal_displacements


if __name__ == "__main__":
    main()
