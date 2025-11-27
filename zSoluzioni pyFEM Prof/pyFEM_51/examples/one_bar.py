#!/usr/bin/env python
"""
Solve a bar in tension.

Created: 2025/10/18 18:18:18
Last modified: 2025/11/17 23:37:58
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def main() -> np.ndarray:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    bar_length = 12.0
    num_nodes = 2
    num_elements = 1

    # Nodal coordinates
    points = np.array([0.0, bar_length])

    # Element connectivity (which nodes belong to each element)
    element_connectivity = [
        [0, 1],
    ]

    # 2. Materials
    materials = pyfem.make_materials(
        [
            ("marble", pyfem.LinearElastic1D(E=23.2)),
        ]
    )

    # 3. Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            (
                "bar",
                pyfem.ElementProperty(
                    "bar_1D", {"A": 7.0}, material="marble", meta={"integration": 3}
                ),
            ),
        ]
    )

    # Assign properties to elements
    element_property_labels = ["bar"]

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
    mesh.add_node_set(tag=2, nodes={1}, name="right_end")

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
    model.bc.apply_force("right_end", pyfem.DOFType.U_X, 10.0)

    # PROCESSING: Solve FEA problem

    # Initialize model state
    model_state = pyfem.ModelState()

    # Define step
    step = pyfem.Step(
        name="StaticLoad",
        procedure=pyfem.ProcedureType.STATIC_LINEAR,
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

    return model_state.current_solution.nodal_displacements


if __name__ == "__main__":
    main()
