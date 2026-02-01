#!/usr/bin/env python
"""
Transient 1D heat diffusion using explicit (forward Euler) time integration.

This example solves:

    ∂T/∂t = α ∂²T/∂x² + s(x)

with Dirichlet boundary conditions and optional internal heat source.
The discretization uses 1D linear diffusion elements ("bar_1D_heat"),
a lumped mass matrix, and a forward-Euler update:

    T^{n+1} = T^n + dt * M_lumped^{-1} (F^n - K T^n)

Notes:
    - Governing physics is Physics.HEAT_TRANSFER.
    - DOFType.TEMPERATURE is used for nodal temperature unknowns.
    - Stability automatically checks the explicit condition:

            Fo = α dt / dx² ≤ 0.5

      and raises a ValueError if violated.

Created: 2025/12/08 16:15:24
Last modified: 2026/01/02 15:26:37
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def main() -> np.ndarray:
    # PREPROCESSING

    # 1. Geometry and discretization
    bar_length = 10000.0  # [m]

    # Use more than 2 nodes so diffusion is actually visible
    num_nodes = 41
    num_elements = num_nodes - 1

    # Nodal coordinates: uniform mesh
    points = np.linspace(0.0, bar_length, num_nodes)

    # Element connectivity (which nodes belong to each element)
    element_connectivity = [[i, i + 1] for i in range(num_elements)]

    # 2. Materials
    alpha = 1.22e-6
    materials = pyfem.make_materials(
        [
            #  alpha: thermal diffusivity [m²/s]
            ("mate1", pyfem.Diffusion1D(alpha=alpha)),
        ]
    )

    # 3. Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            (
                "bar_heat",
                pyfem.ElementProperty(
                    kind="bar_1D_heat",
                    params={"source": 9.6296e-13},  # °C/s
                    material="mate1",
                ),
            ),
        ]
    )

    # Assign properties to elements
    element_property_labels = ["bar_heat"] * num_elements

    # 4. Mesh

    mesh = pyfem.Mesh(
        num_nodes=num_nodes,
        points=points,
        num_elements=num_elements,
        element_connectivity=element_connectivity,
        element_property_labels=element_property_labels,
    )

    # Define node sets for boundary conditions
    mesh.add_node_set(tag=1, nodes={0}, name="left_end")
    mesh.add_node_set(tag=2, nodes={num_nodes - 1}, name="right_end")

    print("\n- Node sets:")
    for tag, node_set in mesh.node_sets.items():
        print(f"  {node_set}")

    # 5. Create Model
    problem = pyfem.Problem(
        pyfem.Physics.HEAT_TRANSFER,
        pyfem.Dimension.D1,
    )

    model = pyfem.Model(mesh, problem)
    model.set_materials(materials)
    model.set_element_properties(element_properties)
    print(model)

    # Initial temperature T0(x)
    ndofs = model.dof_space.total_dofs
    T0 = np.zeros(ndofs, dtype=float)

    for node, x in enumerate(points):
        dof = model.dof_space.get_global_dof(node, pyfem.DOFType.TEMPERATURE)
        T0[dof] = 0.0

        # Bar initially cold, except middle node at 50
        # if node == len(points) // 2:
        #     T0[dof] = 50.0
        # else:
        #     T0[dof] = 0.0

    model.set_initial_temperature(T0)

    # 6. Boundary conditions

    # Dirichlet BCs: prescribed temperatures
    # T(0,t) = 0, T(L,t) = 0
    model.bc.prescribe_dirichlet("left_end", pyfem.DOFType.TEMPERATURE, 0.0)
    model.bc.prescribe_dirichlet("right_end", pyfem.DOFType.TEMPERATURE, 0.0)

    # No Neumann forces here; for heat, they'd correspond to flux boundary conditions.
    # model.bc.apply_force(...)

    model.bc.print_summary()

    # PROCESSING: Solve transient diffusion problem

    # Time discretization parameters
    increments = 400  # number of time steps
    total_time = 50000.0 * 365 * 24 * 60 * 60  # total physical time in seconds

    # STABILITY CHECK FOR EXPLICIT SCHEME (Fourier number)
    #
    # Stability condition for 1D explicit heat equation:
    #   Fo = alpha * dt / dx^2 <= 0.5
    #
    #   Fo_max is the maximum Fourier number (often simply called the Fo number)
    #   associated with your chosen time step and spatial discretization.
    #
    # For non-uniform meshes we use the smallest element length.

    dx_all = np.diff(points)
    dx_min = dx_all.min()
    dt = total_time / increments
    dt_crit = dx_min**2 / (2.0 * alpha)
    Fo_max = alpha * dt / dx_min**2

    print("\n- Explicit time step (informative check):")
    print(f"  dx_min   = {dx_min:.6e} m")
    print(f"  alpha    = {alpha:.6e} m^2/s")
    print(f"  dt       = {dt:.6e} s")
    print(f"  dt_crit  = {dt_crit:.6e} s")
    print(f"  Fo_max   = {Fo_max:.6f}")

    # ------------------------------------------------------------------

    # Initialize model state
    model_state = pyfem.ModelState()

    # Define explicit diffusion step
    step = pyfem.Step(
        name="HeatExplicit",
        procedure=pyfem.ProcedureType.DIFFUSION_EXPLICIT,
        increments=increments,
        total_time=total_time,
        verbose=True,
        output_fields=["temperature"],
        output_frequency=20,  # dump every 20th time step
        output_save_to="vtk_heat_results",  # folder
        output_file="heat_step.vtu",  # base name
    )

    # use the warp by scalar filter in paraview to see the temperature profile

    # Execute step (returns updated state)
    model_state = step.execute(model, model_state, use_sparse=False)

    # POSTPROCESSING: here we simply print final temperature field

    T_final = model_state.current_solution.temperature

    print("\nFinal nodal temperatures:")
    for i, (x, T) in enumerate(zip(points, T_final)):
        print(f"  Node {i:2d} at x = {x:6.3f}  ->  T = {T:10.4f}")

    return T_final


if __name__ == "__main__":
    main()
