#!/usr/bin/env python
"""
Patch test for tetrahedral elements: Steel cube under uniform pressure.

The problem is simplified using symmetry, analyzing only one quarter of the
2 cm x 2 cm x 2 cm cube (i.e., 1 cm x 1 cm x 2 cm) with tetrahedral elements.

Material properties:
- Young's modulus: 210 GPa = 2.1e5 N/mm^2
- Poisson's ratio: 0.3

Applied load:
- Uniform pressure: p = 300 MPa = 300 N/mm^2

Units: mm, N, MPa = N/mm^2

Created: 2025/11/22 23:16:15
Last modified: 2025/11/24 01:28:10
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def main() -> np.ndarray:
    # PREPROCESSING

    # 1. Geometry and discretization

    # Problem parameters
    num_nodes = 8
    num_elements = 6

    # Nodal coordinates
    points = np.array(
        [
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 20.0],
            [0.0, 10.0, 20.0],
            [0.0, 0.0, 20.0],
            [10.0, 0.0, 20.0],
        ]
    )

    # Element connectivity (which nodes belong to each element)
    element_connectivity = [
        [0, 1, 2, 6],
        [0, 5, 1, 6],
        [0, 4, 5, 6],
        [0, 7, 4, 6],
        [0, 3, 7, 6],
        [0, 2, 3, 6],
    ]

    # 2. Materials
    materials = pyfem.make_materials(
        [
            (
                "steel",
                pyfem.LinearElastic3D(
                    E=2.1e5,
                    nu=0.3,
                ),
            ),
        ]
    )

    # 3. Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            ("t4", pyfem.ElementProperty("tetra", material="steel")),
        ]
    )

    # Assign properties to elements
    element_property_labels = ["t4"] * num_elements

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
        pyfem.Dimension.D3,
    )

    model = pyfem.Model(mesh, problem)
    model.set_materials(materials)
    model.set_element_properties(element_properties)
    print(model)

    # 6. Boundary conditions

    # Dirichlet boundary conditions (prescribed displacements)
    model.bc.prescribe_displacement(0, pyfem.DOFType.U_Z, 0.0)
    model.bc.prescribe_displacement(1, pyfem.DOFType.U_Z, 0.0)
    model.bc.prescribe_displacement(2, pyfem.DOFType.U_Z, 0.0)
    model.bc.prescribe_displacement(3, pyfem.DOFType.U_Z, 0.0)
    model.bc.prescribe_displacement(1, pyfem.DOFType.U_X, 0.0)
    model.bc.prescribe_displacement(2, pyfem.DOFType.U_X, 0.0)
    model.bc.prescribe_displacement(5, pyfem.DOFType.U_X, 0.0)
    model.bc.prescribe_displacement(6, pyfem.DOFType.U_X, 0.0)
    model.bc.prescribe_displacement(2, pyfem.DOFType.U_Y, 0.0)
    model.bc.prescribe_displacement(3, pyfem.DOFType.U_Y, 0.0)
    model.bc.prescribe_displacement(6, pyfem.DOFType.U_Y, 0.0)
    model.bc.prescribe_displacement(7, pyfem.DOFType.U_Y, 0.0)

    # Neumann boundary conditions (applied forces)
    model.bc.apply_force(4, pyfem.DOFType.U_Z, -1e4)
    model.bc.apply_force(5, pyfem.DOFType.U_Z, -5e3)
    model.bc.apply_force(6, pyfem.DOFType.U_Z, -1e4)
    model.bc.apply_force(7, pyfem.DOFType.U_Z, -5e3)

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
        operations=["element_stresses"],
    )

    pyfem.VTKWriter(model, model_state.current_solution).write(
        "patch_test_tetra.vtu",
        displacements=True,
        stresses=True,
    )

    print(model_state.current_solution.element_stresses)

    return model_state.current_solution.nodal_displacements


if __name__ == "__main__":
    main()
