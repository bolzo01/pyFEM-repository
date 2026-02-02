#!/usr/bin/env python
"""
Transient 2D heat diffusion using explicit (forward Euler) time integration.

This example solves:

    ρ Cp (∂T/∂t) = ∇(D∇T) + s

with Dirichlet boundary conditions and optional internal heat source.
The discretization uses 2D linear diffusion elements ("heat_exchanger_2d_heat"),
a lumped mass matrix, and a forward-Euler update:

    T^{n+1} = T^n + dt * M_lumped^{-1} (F^n - K T^n)

Notes:
    - Governing physics is Physics.HEAT_TRANSFER.
    - DOFType.TEMPERATURE is used for nodal temperature unknowns.
    - Stability automatically checks the explicit condition:

            Fo = α dt / dx² ≤ 0.5

      and raises a ValueError if violated.

Created: 2025/12/08 16:15:24
Last modified: 2026/02/01 19:06:06
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def main() -> np.ndarray:
    # PREPROCESSING

    # 1. Build mesh from Gmsh
    mesh = pyfem.Mesh.from_gmsh("heat_exchanger.msh", dim=2)

    # 2. Materials
    # rho   = density [kg/m³]
    # c     = specific heat capacity [J/(kg*K)]
    # k     = thermal conductivity [W/(m*K)]
    # alpha = k / (rho * c) = thermal diffusivity [m²/s]

    # Steel  -> AISI 316L (EN 1.4404)
    # Copper -> Copper DHP (UNS C12200 / CW024A)
    # Water  -> tap water

    materials = pyfem.make_materials(
        [
            (
                "steel",
                pyfem.Diffusion2D(
                    rho=8000.0, c=500.0, k=15.0, alpha=(15.0 / (8000.0 * 500.0))
                ),
            ),
            (
                "water",
                pyfem.Diffusion2D(
                    rho=1000.0, c=4182.0, k=0.6, alpha=(0.6 / (1000.0 * 4184.0))
                ),
            ),
        ]
    )

    # 3. Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            (
                "matrix",
                pyfem.ElementProperty(
                    kind="triangle_heat",
                    params={"source": 0.0, "t": 1.0},  # °C/s , m
                    material="steel",
                ),
            ),
        ]
    )

    element_properties = pyfem.make_element_properties(
        [
            (
                "tube_hot",
                "tube_cold",
                pyfem.ElementProperty(
                    kind="triangle_heat",
                    params={"source": 0.0, "t": 1.0},  # °C/s , m
                    material="water",
                ),
            ),
        ]
    )

    # 4. Create Model
    problem = pyfem.Problem(
        pyfem.Physics.HEAT_TRANSFER,
        pyfem.Dimension.D2,
    )
    model = pyfem.Model(mesh, problem)
    model.set_materials(materials)
    model.set_element_properties(element_properties)
    print(model)

    # Initial temperature T0(x)

    ndofs = model.dof_space.total_dofs
    points = mesh.points
    T0 = np.zeros(ndofs, dtype=float)

    for node, coords in enumerate(points):
        dof = model.dof_space.get_global_dof(node, pyfem.DOFType.TEMPERATURE)

        # 1. Extract actual node coordinates x and y
        x_node, y_node = coords[0], coords[1]

        # 2. Distance from tube centers
        # 2nd tube centered at (10.0, 3.33) with radius 2.0
        dist_tube_1 = np.sqrt((x_node - 10.0) ** 2 + (y_node - 3.33) ** 2)
        # 4th tube centered at (15.0, 6.66) with radius 2.0
        dist_tube_4 = np.sqrt((x_node - 15.0) ** 2 + (y_node - 6.66) ** 2)
        # 6th tube centered at (5.0, 6.66) with radius 2.0
        dist_tube_6 = np.sqrt((x_node - 5.0) ** 2 + (y_node - 3.33) ** 2)

        # 3. Set initial temperature based on distance from tube centers
        tube_radius = 2.0
        if (
            (dist_tube_1 <= tube_radius)
            or (dist_tube_4 <= tube_radius)
            or (dist_tube_6 <= tube_radius)
        ):
            T0[dof] = 80.0  # Hot tube temperature
        else:
            T0[dof] = 20.0  # Matrix and cold tube temperature

    model.set_initial_temperature(T0)

    # 6. Boundary conditions

    # Dirichlet BCs: prescribed temperatures
    model.bc.prescribe_dirichlet("cold_surface", pyfem.DOFType.TEMPERATURE, 20.0)
    model.bc.prescribe_dirichlet("hot_surface", pyfem.DOFType.TEMPERATURE, 80.0)

    # No Neumann flux boundary conditions here.

    model.bc.print_summary()

    # PROCESSING: Solve transient diffusion problem

    # Time discretization parameters
    increments = 400  # number of time steps
    total_time = 60 * 5  # total physical time in seconds

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
        output_file="heat_step_steel.vtu",  # base name
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
