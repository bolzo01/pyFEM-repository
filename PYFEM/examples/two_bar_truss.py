#!/usr/bin/env python
"""
Two-bar truss - Example 2 in Trusses.pdf.

Created: 2025/10/31 14:34:37
Last modified: 2025/11/18 00:17:49
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
    points = np.array(
        [
            [0.0, 0.0],
            [0.5 * np.sqrt(2), 0.5 * np.sqrt(2)],
            [0.0, np.sqrt(2)],
        ]
    )

    # Element connectivity (which nodes belong to each element)
    element_connectivity = [
        [0, 1],
        [1, 2],
    ]

    # 2. Materials
    materials = pyfem.make_materials(
        [
            ("mate1", pyfem.LinearElastic1D(E=3)),
        ]
    )

    # 3. Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            (
                "bar",
                pyfem.ElementProperty("bar_2D", {"A": 2.0}, material="mate1"),
            ),
        ]
    )

    # Assign properties to elements
    element_property_labels = ["bar"] * 2

    # 4. Mesh

    # Create mesh
    mesh = pyfem.Mesh(
        num_nodes=num_nodes,
        points=points,
        num_elements=num_elements,
        element_connectivity=element_connectivity,
        element_property_labels=element_property_labels,
    )

    # 5. Create Model

    problem = pyfem.Problem(
        pyfem.Physics.MECHANICS,
        pyfem.Dimension.D2,
    )

    model = pyfem.Model(mesh, problem)
    model.set_materials(materials)
    model.set_element_properties(element_properties)
    print(model)

    # 6. Boundary conditions

    # Dirichlet boundary conditions (prescribed displacements)
    model.bc.prescribe_displacement(0, pyfem.DOFType.U_X, 0.0)
    model.bc.prescribe_displacement(0, pyfem.DOFType.U_Y, 0.0)
    model.bc.prescribe_displacement(2, pyfem.DOFType.U_X, 0.0)
    model.bc.prescribe_displacement(2, pyfem.DOFType.U_Y, 0.0)

    # Neumann boundary conditions (applied forces)
    model.bc.apply_force(1, pyfem.DOFType.U_X, 1.0)
    model.bc.apply_force(1, pyfem.DOFType.U_Y, 2.0)

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

    # Reactions
    solution.print_reactions(model.dof_space)

    # Check global equilibrium
    solution.check_equilibrium(solver, tolerance=1e-8, verbose=True)

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
