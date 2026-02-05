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
Last modified: 2026/02/05 18:38:43
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem
from pyfem.solvers import DiffusionExplicitSolver


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
            (
                "tube_hot",
                pyfem.ElementProperty(
                    kind="triangle_heat",
                    params={"source": 0.0, "t": 1.0},  # °C/s , m
                    material="water",
                ),
            ),
            (
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
        dist_tube_2 = np.sqrt((x_node - 10.0) ** 2 + (y_node - 3.33) ** 2)
        # 4th tube centered at (15.0, 6.66) with radius 2.0
        dist_tube_4 = np.sqrt((x_node - 15.0) ** 2 + (y_node - 6.66) ** 2)
        # 6th tube centered at (5.0, 6.66) with radius 2.0
        dist_tube_6 = np.sqrt((x_node - 5.0) ** 2 + (y_node - 6.66) ** 2)

        # 3. Set initial temperature based on distance from tube centers
        tube_radius = 1.0
        if (
            (dist_tube_2 <= tube_radius)
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
    total_time = 60.0 * 60.0 * 10.0  # total physical time in seconds

    # STABILITY CHECK FOR EXPLICIT SCHEME (Fourier number)
    #
    solver = DiffusionExplicitSolver(model)
    dt_crit = solver.compute_critical_timestep_2D()
    increments = int(total_time / dt_crit)
    is_stable = solver.check_stability(dt_crit, verbose=True)
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
        output_frequency=int(increments / 20.0),  # dump every nth time step
        output_save_to="vtk_heat_results_steel",  # folder
        output_file="heat_step_steel.vtu",  # base name
    )

    # use the warp by scalar filter in paraview to see the temperature profile

    # Execute step (returns updated state)
    model_state = step.execute(model, model_state)

    # POSTPROCESSING: statistics filtered only for the matrix region

    T_final = model_state.current_solution.temperature

    # Let's find the indices of the elements that are labeled as "matrix"
    #    mesh.element_property_labels is a list that says "matrix", "tube_hot", ecc. for each element
    matrix_element_indices = [
        i for i, label in enumerate(mesh.element_property_labels) if label == "matrix"
    ]

    # Let's find the unique nodes that belong to these "matrix" elements
    #    set counteracts duplications (each node is shared by multiple triangles)
    matrix_nodes = set()
    for elem_idx in matrix_element_indices:
        nodes = mesh.element_connectivity[elem_idx]
        matrix_nodes.update(nodes)

    # List conversion for indexing
    matrix_nodes_list = list(matrix_nodes)

    # Let's extract the T values only for these nodes
    T_matrix = T_final[matrix_nodes_list]

    # Let's print the results
    print("\n" + "=" * 40)
    print(f"RESULTS FOR REGION: MATRIX ({element_properties['matrix'].material})")
    print("=" * 40)
    print(f"  Nodes in matrix:            {len(T_matrix)} (out of {len(T_final)})")
    print(f"  Min Temperature:  {np.min(T_matrix):.4f} °C")
    print(f"  Max Temperature:  {np.max(T_matrix):.4f} °C")
    print(f"  Avg Temperature:  {np.mean(T_matrix):.4f} °C")
    print(f"  Total time (min): {total_time / 60} min")
    print(
        "  Note: Tubes and water regions are excluded from these stats because the T is constant."
    )
    print("-" * 40)
    print("  Full field results saved to: 'vtk_heat_results/*.vtu'")
    print("  Open with Paraview to visualize.")
    print("=" * 40 + "\n")

    return T_final


if __name__ == "__main__":
    main()
