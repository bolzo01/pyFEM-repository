#!/usr/bin/env python
"""
Series combination of one spring and two bars in tension.
Uses the new Step + ModelState architecture.

Created: 2025/10/18 22:16:45
Last modified: 2025/11/17 22:03:10
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def single_step_analysis():
    """Linear static analysis using Step and ModelState pattern."""

    # PREPROCESSING: Define the model

    print("=" * 70)
    print("ONE SPRING + TWO BARS IN SERIES")
    print("=" * 70)

    # Mesh definition
    mesh = pyfem.Mesh(
        num_nodes=4,
        points=np.array([0.0, 1.0, 2.0, 3.0]),
        num_elements=3,
        element_connectivity=[[0, 1], [1, 2], [2, 3]],
        element_property_labels=["spring", "thick_bar", "thin_bar"],
    )

    # Problem definition
    problem = pyfem.Problem(pyfem.Physics.MECHANICS, pyfem.Dimension.D1)

    # Create model
    model = pyfem.Model(mesh, problem)

    # Materials
    materials = pyfem.make_materials(
        [
            ("mate1", pyfem.LinearElastic1D(E=2.0)),
        ]
    )

    # Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            (
                "thick_bar",
                pyfem.ElementProperty("bar_1D", {"A": 2.0}, material="mate1"),
            ),
            (
                "thin_bar",
                pyfem.ElementProperty("bar_1D", {"A": 1.0}, material="mate1"),
            ),
            (
                "spring",
                pyfem.ElementProperty(kind="spring_1D", params={"k": 2.0}),
            ),
        ]
    )

    model.set_materials(materials)
    model.set_element_properties(element_properties)

    # Boundary conditions
    model.bc.prescribe_displacement(0, pyfem.DOFType.U_X, 0.0)  # Fixed left end
    model.bc.apply_force(3, pyfem.DOFType.U_X, 4.0)  # Applied force

    # ANALYSIS: Execute step

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
        operations=[
            "strain_energy_local",
            "equilibrium_check",
        ],
    )

    # Manual analysis
    print("\n" + "=" * 70)
    print("DETAILED SOLUTION ANALYSIS")
    print("=" * 70)

    solution = model_state.current_solution

    # Print displacements
    print("\nNodal Displacements:")
    for node in range(len(solution.nodal_displacements)):
        dof = model.dof_space.get_global_dof(node, pyfem.DOFType.U_X)
        u = solution.nodal_displacements[dof]
        print(f"  Node {node}: u_x = {u:.6e} m")

    # Print reactions
    print("\nReaction Forces:")
    for node in range(len(solution.nodal_displacements)):
        dof = model.dof_space.get_global_dof(node, pyfem.DOFType.U_X)
        reaction = solution.get_reaction_force(dof)
        if abs(reaction) > 1e-10:
            print(f"  Node {node}: R_x = {reaction:.6f} N")
        else:
            print(f"  Node {node}: (free, no reaction)")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    print(model_state)

    return model_state


def multi_step_analysis():
    """Example: Multiple load cases (Abaqus multi-step pattern)."""

    print("\n" + "=" * 70)
    print("MULTI-STEP ANALYSIS: TWO LOAD CASES")
    print("=" * 70)

    # Setup model
    mesh = pyfem.Mesh(
        num_nodes=4,
        points=np.array([0.0, 1.0, 2.0, 3.0]),
        num_elements=3,
        element_connectivity=[[0, 1], [1, 2], [2, 3]],
        element_property_labels=["spring", "thick_bar", "thin_bar"],
    )

    problem = pyfem.Problem(pyfem.Physics.MECHANICS, pyfem.Dimension.D1)
    model = pyfem.Model(mesh, problem)

    materials = pyfem.make_materials(
        [
            ("mate1", pyfem.LinearElastic1D(E=2.0)),
        ]
    )

    element_properties = pyfem.make_element_properties(
        [
            (
                "thick_bar",
                pyfem.ElementProperty("bar_1D", {"A": 2.0}, material="mate1"),
            ),
            (
                "thin_bar",
                pyfem.ElementProperty("bar_1D", {"A": 1.0}, material="mate1"),
            ),
            (
                "spring",
                pyfem.ElementProperty(kind="spring_1D", params={"k": 2.0}),
            ),
        ]
    )

    model.set_materials(materials)
    model.set_element_properties(element_properties)

    # Initialize state
    model_state = pyfem.ModelState()

    # -------------------------------------------------------------------------
    # STEP 1: First load case
    # -------------------------------------------------------------------------
    print("\n--- Configuring Step 1 ---")
    model.bc.prescribe_displacement(0, pyfem.DOFType.U_X, 0.0)
    model.bc.apply_force(3, pyfem.DOFType.U_X, 4.0)

    step1 = pyfem.Step(name="LoadCase1", procedure=pyfem.ProcedureType.STATIC_LINEAR)
    model_state = step1.execute(model, model_state, use_sparse=False)

    # -------------------------------------------------------------------------
    # STEP 2: Second load case (different force, twice the previous value)
    # -------------------------------------------------------------------------
    print("\n--- Configuring Step 2 ---")
    # Clear previous forces
    model.bc.registry._forces.clear()
    model.bc.apply_force(3, pyfem.DOFType.U_X, 8.0)  # Double the force

    step2 = pyfem.Step(name="LoadCase2", procedure=pyfem.ProcedureType.STATIC_LINEAR)
    model_state = step2.execute(model, model_state, use_sparse=False)

    # -------------------------------------------------------------------------
    # COMPARE RESULTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON OF LOAD CASES")
    print("=" * 70)

    sol1 = model_state.get_step_solution("LoadCase1")
    sol2 = model_state.get_step_solution("LoadCase2")

    u1_max = np.max(np.abs(sol1.nodal_displacements))
    u2_max = np.max(np.abs(sol2.nodal_displacements))

    print("\nLoadCase1 (F=4.0 N):")
    print(f"  Max displacement: {u1_max:.6e} m")
    print(f"  Total reaction:   {sol1.get_reaction_force(0):.6f} N")

    print("\nLoadCase2 (F=8.0 N):")
    print(f"  Max displacement: {u2_max:.6e} m")
    print(f"  Total reaction:   {sol2.get_reaction_force(0):.6f} N")

    print(f"\nRatio (should be 2.0): {u2_max / u1_max:.6f}")

    print("\n" + "=" * 70)
    print(model_state)
    print("=" * 70)

    return model_state


if __name__ == "__main__":
    # Run single-step analysis
    state_single = single_step_analysis()

    # Run multi-step analysis
    state_multi = multi_step_analysis()
