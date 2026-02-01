#!/usr/bin/env python
"""
Patch test for CST triangular elements.

Problem 7.20 in "The Finite Element Method: Its Basis and Fundamentals", Seventh Edition, by
Zienkiewicz, Taylor, Zhu (2013).

Created: 2025/11/21 18:04:29
Last modified: 2025/11/27 12:26:27
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def main() -> np.ndarray:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    num_nodes = 4
    num_elements = 2

    # Nodal coordinates
    points = np.array(
        [
            [0.0, 0.0],
            [6.0, 0.0],
            [6.0, 5.0],
            [0.0, 5.0],
        ]
    )

    # Element connectivity (which nodes belong to each element)
    element_connectivity = [
        [0, 1, 3],
        [1, 2, 3],
    ]

    # 2. Materials
    materials = pyfem.make_materials(
        [
            (
                "mate1",
                pyfem.LinearElastic2D(
                    E=200000.0,
                    nu=0.3,
                    formulation="plane_stress",
                ),
            ),
        ]
    )

    # 3. Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            (
                "t3",
                pyfem.ElementProperty(
                    "triangle",
                    {"t": 1.0},
                    material="mate1",
                ),
            ),
        ]
    )

    # Assign properties to elements
    element_property_labels = ["t3"] * num_elements

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
    model.bc.prescribe_displacement(1, pyfem.DOFType.U_Y, 0.0)
    model.bc.prescribe_displacement(3, pyfem.DOFType.U_X, 0.0)

    # Neumann boundary conditions (applied forces)
    model.bc.apply_force(1, pyfem.DOFType.U_X, 5.0)
    model.bc.apply_force(2, pyfem.DOFType.U_X, 5.0)

    # PROCESSING: Solve FEA problem

    # Initialize model state
    model_state = pyfem.ModelState()

    # Define step
    step = pyfem.Step(
        name="ramp test",
        procedure=pyfem.ProcedureType.STATIC_LINEAR_INCREMENTAL,
        control="force",
        increments=10,
        # ramp(i,n): i=current step, n=total steps
        # ramp=lambda i, n: i/n          # linear
        # The function must return a load factor between 0 and 1.
        # this is a quadratic ramp (slow at beginning, fast towards the end)
        ramp=lambda i, n: (i / n) ** 2,
        output_fields=["displacements", "element_stresses"],
        output_frequency=2,
        output_save_to="vtk_results",
        output_file="load_step.vtu",
        verbose=True,
    )

    # Execute step (returns updated state)
    model_state = step.execute(model, model_state, use_sparse=False)

    # POSTPROCESSING: Analyze results

    # Post-process using the built-in method of the step
    step.postprocess(
        model,
        model_state,
        operations=["element_stresses"],
    )

    model.save_vtk(
        filename="patch_test_triangle.vtu",
        displacements=True,
        stresses=True,
        save_to="vtk_results",
    )

    print(model_state.current_solution.element_stresses)

    return model_state.current_solution.nodal_displacements


if __name__ == "__main__":
    main()
