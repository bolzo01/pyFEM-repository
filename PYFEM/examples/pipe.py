#!/usr/bin/env python
"""
A porous lattice structure with mesh from SLT file.

Created: 2025/11/25 01:26:45
Last modified: 2025/11/26 03:03:54
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import pyfem


def main():
    # Build mesh from Gmsh
    mesh = pyfem.Mesh.from_gmsh("pipe.msh", dim=3)

    materials = pyfem.make_materials(
        [("steel", pyfem.LinearElastic3D(E=2.1e5, nu=0.3))]
    )

    # Element properties: region names must match those from Gmsh
    element_properties = pyfem.make_element_properties(
        [
            ("solid", pyfem.ElementProperty("tetra", material="steel")),
        ]
    )

    problem = pyfem.Problem(pyfem.Physics.MECHANICS, pyfem.Dimension.D3)
    model = pyfem.Model(mesh, problem)
    model.set_materials(materials)
    model.set_element_properties(element_properties)

    # Coordinate-based boundary conditions

    # - Constrain bottom surface (z<-0.5)
    model.bc.prescribe_where(lambda x: x[2] < -0.5, pyfem.DOFType.U_X, 0.0)
    model.bc.prescribe_where(lambda x: x[2] < -0.5, pyfem.DOFType.U_Y, 0.0)
    model.bc.prescribe_where(lambda x: x[2] < -0.5, pyfem.DOFType.U_Z, 0.0)

    # - Prescribed displacement on top surface
    cx, cy = 20.0, 20.0  # coordinates of the circle center
    radius = 3.5  # circle radius
    model.bc.prescribe_where(
        lambda x: (x[2] > 28.0) and ((x[0] - cx) ** 2 + (x[1] - cy) ** 2 <= radius**2),
        pyfem.DOFType.U_Z,
        -10.0,
    )

    # # -- Alternative definition:
    # def in_displaced_volume(x):
    #     is_on_top = x[2] > 28.0
    #     is_inside = (x[0] - cx) ** 2 + (x[1] - cy) ** 2 <= radius**2
    #     return is_on_top and is_inside

    # model.bc.prescribe_where(in_displaced_volume, pyfem.DOFType.U_Z, -10.0)

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
    model_state = step.execute(model, model_state)

    # POSTPROCESSING: Analyze results

    step.postprocess(
        model,
        model_state,
        operations=["element_stresses"],
    )

    model.save_vtk(
        "pipe.vtu",
        displacements=True,
        stresses=True,
        save_to="vtk_results",
    )
    # model.save_vtk( ) : saves next to the running example
    # model.save_vtk(save_to="vtk_results") : saves into a user-defined subfolder
    # model.save_vtk("/tmp/test.vtu") : saves to an absolute path
    # creates folder if folder doesn't exist

    return model_state.current_solution.nodal_displacements


if __name__ == "__main__":
    main()
