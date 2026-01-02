#!/usr/bin/env python
"""
A square domain with circular inclusions under tension.

Created: 2025/11/25 01:11:13
Last modified: 2025/11/27 14:25:36
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import pyfem


def main():
    # Build mesh from Gmsh
    mesh = pyfem.Mesh.from_gmsh("circular_inclusions.msh", dim=2)

    # Materials
    materials = pyfem.make_materials(
        [
            (
                "mate1",
                pyfem.LinearElastic2D(E=1.0, nu=0.2, formulation="plane_strain"),
            ),
            (
                "mate2",
                pyfem.LinearElastic2D(E=100.0, nu=0.3, formulation="plane_strain"),
            ),
        ]
    )

    # Element properties: region names must match those from Gmsh
    element_properties = pyfem.make_element_properties(
        [
            (
                "matrix",
                pyfem.ElementProperty("triangle", {"t": 1.0}, material="mate1"),
            ),
            (
                "inclusion",
                pyfem.ElementProperty("triangle", {"t": 1.0}, material="mate2"),
            ),
        ]
    )

    problem = pyfem.Problem(pyfem.Physics.MECHANICS, pyfem.Dimension.D2)
    model = pyfem.Model(mesh, problem)
    model.set_materials(materials)
    model.set_element_properties(element_properties)

    # Boundary conditions using node set names from Gmsh
    model.bc.prescribe_displacement("left side", pyfem.DOFType.U_X, 0.0)
    model.bc.prescribe_displacement("right side", pyfem.DOFType.U_X, 1.0)
    model.bc.prescribe_displacement("bottom left corner", pyfem.DOFType.U_Y, 0.0)

    # PROCESSING: Solve FEA problem

    # Initialize model state
    model_state = pyfem.ModelState()

    # Define step
    step = pyfem.Step(
        name="Incremental displacements",
        procedure=pyfem.ProcedureType.STATIC_LINEAR_INCREMENTAL,
        control="displacement",
        increments=20,
        # ramp(i,n): i=current step, n=total steps
        # ramp=lambda i, n: i/n          # linear
        # The function must return a load factor between 0 and 1.
        # this is a quadratic ramp (slow at beginning, fast towards the end)
        ramp=lambda i, n: i / n,
        output_fields=["displacements", "element_stresses"],
        output_frequency=2,
        output_save_to="vtk_results",
        output_file="load_step.vtu",
        verbose=True,
    )

    # Execute step (returns updated state)
    model_state = step.execute(model, model_state)

    # POSTPROCESSING: Analyze results

    step.postprocess(
        model,
        model_state,
        operations=["element_stresses"],
    )

    model.save_vtk(
        filename="circular_inclusion.vtu",
        displacements=True,
        stresses=True,
        save_to="vtk_results",
    )

    # # Options:
    # # Default behavior (save next to the running example)
    # model.save_vtk(
    #     filename="circular_inclusion.vtu",
    #     displacements=True,
    #     stresses=True,
    # )

    # # Save inside a relative subfolder
    # model.save_vtk(
    #     filename="circular_inclusion.vtu",
    #     displacements=True,
    #     stresses=True,
    #     save_to="vtk_results",  # folder vtk_results created automatically
    # )

    # # Or provide an absolute path manually
    # model.save_vtk(
    #     filename="/tmp/test.vtu",
    #     displacements=True,
    #     stresses=True,
    # )

    return model_state.current_solution.nodal_displacements


if __name__ == "__main__":
    main()
