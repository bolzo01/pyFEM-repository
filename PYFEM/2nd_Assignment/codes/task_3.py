#!/usr/bin/env python

import numpy as np

import pyfem


def genera_connettivita_uniforme(Ne: int) -> list:
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


def main() -> np.ndarray:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    L = 10.0
    number_elements = [4, 8, 16, 32, 64]
    P = 1.0
    E = 1.0
    A = 1.0
    alpha = 1.0

    # 2. Element properties

    # Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            ("bar", ("bar_1D", {"E": 1.0, "A": 1.0, "k": 1.0})),
        ]
    )

    for Ne in number_elements:
        num_elements = Ne

        # Assign properties to elements
        element_property_labels = ["bar"] * num_elements

        # 1. Calcola Nodi e Punti
        num_nodes = Ne + 1
        points = np.linspace(0.0, 10.0, num_nodes)

        # 2. Genera la Connettività
        element_connectivity = genera_connettivita_uniforme(Ne)

        print("elem connec: ", element_connectivity)

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
        nodal_displacements, original_global_stiffness_matrix = solver.solve()

        # POSTPROCESSING: Compute derived quantities

        # Create postprocessor
        postprocessor = pyfem.PostProcessor(
            model.mesh,
            model.element_properties,
            original_global_stiffness_matrix,
            nodal_displacements,
        )

        # Compute strain energy
        postprocessor.compute_strain_energy_global()

    return nodal_displacements


if __name__ == "__main__":
    main()
