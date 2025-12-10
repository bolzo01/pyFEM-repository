#!/usr/bin/env python

import numpy as np

import pyfem


def connectivity_3nodes_per_element(Ne: int) -> list[list[int]]:
    """
    Generates the connectivity matrix for 1D quadratic elements (3 nodes).

    Parameters:
    Ne (int): Number of finite elements in the mesh.

    Returns:
    element_connectivity (list[list[int]]): A matrix of size (n_elements x 3) containing integer node indices.
    """
    # 1. Initialize the matrix
    element_connectivity = []

    # 2. Fill the matrix
    for element in range(Ne):
        start_node = 2 * element
        # Crea una lista per il singolo elemento [n1, n2, n3]
        element_nodes = [start_node, start_node + 1, start_node + 2]

        # Aggiungila alla lista principale
        element_connectivity.append(element_nodes)

    return element_connectivity


def main(use_sparse: bool = True) -> None:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    L = 10.0
    number_elements = np.zeros(18)
    for n in range(0, 18):
        number_elements[n] = 2**n
    h = L / number_elements
    P = 1.0
    E = 1.0
    A = 1.0
    alpha = 1.0

    epsilon = (P / (2.0 * alpha * E * A)) * (-np.exp(-20.0 * alpha) + 1.0)

    # 2. Element properties

    # Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            ("bar", ("bar3_1D", {"E": E, "A": A, "k": alpha**2 * E * A})),
        ]
    )

    # Initialization of the vectors before the for cicle
    epsilon = np.zeros((1, len(number_elements)))
    epsilon_h = np.zeros((1, len(number_elements)))
    e = np.zeros((1, len(number_elements)))

    for Ne in number_elements:
        num_elements = int(Ne)
        # Found the position in number_elements of num_elements
        i = np.where(number_elements == num_elements)[0][0]

        # Assign properties to elements
        element_property_labels = ["bar"] * num_elements

        # Nodes and points calculation
        num_nodes = int((2.0 * Ne) + 1)
        points = np.linspace(0.0, 10.0, num_nodes)

        # Connectivity generation
        element_connectivity = connectivity_3nodes_per_element(int(Ne))

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
        mesh.add_node_set(tag=2, nodes={num_nodes - 1}, name="right_end")

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
            "right_end",
            pyfem.DOFType.U_X,
            (-(P / (E * A * alpha)) * np.exp(-alpha * 10.0)),
        )

        # Neumann boundary conditions (applied forces)
        model.bc.apply_force("left_end", pyfem.DOFType.U_X, -P)

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

        # Create postprocessor
        postprocessor = pyfem.PostProcessor(
            model.mesh,
            model.element_properties,
            solver.global_stiffness_matrix,
            solver.nodal_displacements,
            number_elements,
            alpha,
            E,
            A,
            P,
            magnification_factor=1.0,
        )

        # Compute strain energy using the global solution (U = 0.5 * u^T * K * u)
        epsilon_h = postprocessor.compute_strain_energy_global(i, epsilon_h)

        # Compute the analytical strain energy
        epsilon = postprocessor.compute_strain_energy_analytical(i, epsilon)

        # Computes the relative error in the energy norm
        e = postprocessor.compute_relative_error_in_energy(i, epsilon, epsilon_h, e)

    # Plots the solution of the energy
    postprocessor.plot_solution(epsilon, epsilon_h)
    # Plots the relative error in the energy
    postprocessor.plot_e(e, h)
    # Plots the relative error in the energy
    postprocessor.plot_convergence_rate(e, h)


if __name__ == "__main__":
    main(use_sparse=False)
