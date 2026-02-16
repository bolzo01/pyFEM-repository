#!/usr/bin/env python

import numpy as np
import scipy.sparse.linalg as spla

import pyfem


def connectivity_2nodes_per_element(Ne: int) -> list:
    """
    Genera la connettività per una mesh 1D uniforme con Ne elementi a due nodi.

    Args:
        Ne: Numero di elementi.

    Returns:
        Una lista di liste che definisce la connettività degli elementi.
    """

    # 1. Crea un array di indici da 0 a Ne (i nodi iniziali)
    nodi_iniziali = np.arange(Ne)  # [0, 1, 2, ..., Ne-1]

    # 2. Crea un array per i nodi finali (da 1 a Ne)
    nodi_finali = np.arange(1, Ne + 1)  # [1, 2, 3, ..., Ne]

    # 3. Combina i due array in una lista di coppie (connettività)
    #    numpy.column_stack crea una matrice a due colonne,
    #    poi la convertiamo in una lista di liste.
    element_connectivity = np.column_stack((nodi_iniziali, nodi_finali)).tolist()

    return element_connectivity


def conditioning_numbers_of_K(alpha_counter, cond_numbers, K, i: int) -> np.ndarray:
    """
    Computes the conditioning number of the matrix K.
    It's a matrix and every row is related to a different value of alpha.

    Returns:
        cond_numbers.
    """

    # 1. Calcola l'autovalore massimo (k=1, Largest Magnitude)
    # return_eigenvectors=False risparmia memoria
    max_eig = spla.norm(K, np.inf)

    # 2. Calcola l'autovalore minimo (k=1, Smallest Magnitude)
    # sigma=0 usa lo shift-invert mode che è molto più veloce per trovare valori vicino allo 0

    try:
        min_eig = spla.eigsh(
            K,
            k=1,
            sigma=0,  # Shift-Invert (cerca l'inverso, velocissimo per il minimo)
            which="LM",  # Largest Magnitude dell'inverso
            return_eigenvectors=False,
            tol=1e-2,  # FONDAMENTALE: Ci accontentiamo dell'1% di errore
            ncv=10,  # Aumentiamo i vettori di Lanczos per convergere in meno iterazioni
        )[0]

        # Se min_eig è troppo vicino a zero, evita divisioni assurde
        if abs(min_eig) < 1e-15:
            c = np.inf
        else:
            c = abs(max_eig / min_eig)

    except (RuntimeError, ValueError):
        # Se il calcolo esplode (matrice singolare), il condizionamento è infinito
        c = np.inf

    # 3. Salva il risultato nella matrice e restituisci la matrice
    cond_numbers[alpha_counter, i] = c

    return cond_numbers


def main(use_sparse: bool = False) -> None:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    L = 10.0
    number_elements = np.zeros(13)
    for n in range(0, 13):
        number_elements[n] = 2**n
    h = L / number_elements
    P = 1.0
    E = 1.0
    A = 1.0
    alpha_values = np.array([0.5, 1.0, 2.0, 4.0])

    # Initialization of the vectors before the cicles
    epsilon = np.zeros((len(alpha_values), len(number_elements)))
    epsilon_h = np.zeros((len(alpha_values), len(number_elements)))
    e = np.zeros((len(alpha_values), len(number_elements)))
    cond_numbers = np.zeros((len(alpha_values), len(number_elements)))

    for alpha in alpha_values:
        alpha_counter = np.where(alpha_values == alpha)[0][0]

        # 2. Element properties

        # Define element properties registry
        element_properties = pyfem.make_element_properties(
            [
                ("bar", ("bar_1D", {"E": E, "A": A, "k": alpha**2 * E * A})),
            ]
        )

        for Ne in number_elements:
            num_elements = int(Ne)

            # Found the position in number_elements of num_elements
            i = np.where(number_elements == num_elements)[0][0]

            # Assign properties to elements
            element_property_labels = ["bar"] * num_elements

            # Nodes and points calculation
            num_nodes = int(Ne + 1)
            points = np.linspace(0.0, 10.0, num_nodes)

            # Connectivity generation
            element_connectivity = connectivity_2nodes_per_element(int(Ne))

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
                alpha_counter,
                alpha_values,
                E,
                A,
                P,
            )

            # Compute strain energy using the global solution (U = 0.5 * u^T * K * u)
            epsilon_h = postprocessor.compute_strain_energy_global(i, epsilon_h)

            # Compute the analytical strain energy
            epsilon = postprocessor.compute_strain_energy_analytical(i, epsilon)

            # Computes the relative error in the energy norm
            e = postprocessor.compute_relative_error_in_energy(i, epsilon, epsilon_h, e)

            # Compute the conditioning number of K
            cond_numbers = conditioning_numbers_of_K(
                alpha_counter, cond_numbers, solver.global_stiffness_matrix, i
            )

    # Plots the solution of the energy
    postprocessor.plot_solution(epsilon, epsilon_h)
    # Plots the relative error in the energy
    postprocessor.plot_e(e, h)
    # Plots the relative error in the energy
    postprocessor.plot_convergence_rate(e, h, cond_numbers)


if __name__ == "__main__":
    main(use_sparse=False)
