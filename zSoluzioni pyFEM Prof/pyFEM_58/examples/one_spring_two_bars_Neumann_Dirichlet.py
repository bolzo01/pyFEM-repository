#!/usr/bin/env python
"""
Solve a series combination of one spring and two bars in tension using
incremental Neumann and Dirichlet (mixed) boundary conditions

Created: 2025/11/27 00:09:24
Last modified: 2025/11/28 09:44:42
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
    model.bc.prescribe_displacement(0, pyfem.DOFType.U_X, 0.0)
    model.bc.prescribe_displacement(3, pyfem.DOFType.U_X, 4.0)

    # Neumann boundary conditions (applied forces)
    model.bc.apply_force(2, pyfem.DOFType.U_X, 6.0)

    model.bc.print_summary()

    # PROCESSING: Solve FEA problem

    # Initialize model state
    model_state = pyfem.ModelState()

    # Define step
    step = pyfem.Step(
        name="Incremental mixed",
        procedure=pyfem.ProcedureType.STATIC_LINEAR_INCREMENTAL,
        control="mixed",
        increments=3,
        # ramp(i,n): i=current step, n=total steps
        # ramp=lambda i, n: i/n          # linear
        # The function must return a load factor between 0 and 1.
        # this is a quadratic ramp (slow at beginning, fast towards the end)
        ramp=lambda i, n: i / n,
        verbose=True,
    )

    # Execute step (returns updated state)
    model_state = step.execute(model, model_state, use_sparse=False)

    # POSTPROCESSING: Analyze results

    # Post-process using the built-in method of the step
    step.postprocess(
        model,
        model_state,
        operations=["strain_energy_local", "element_stresses"],
    )

    print("stress", model_state.current_solution.element_stresses)

    print("displacement", model_state.current_solution.nodal_displacements)

    return model_state.current_solution.nodal_displacements


if __name__ == "__main__":
    main()
