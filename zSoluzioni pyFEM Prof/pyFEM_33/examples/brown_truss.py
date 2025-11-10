#!/usr/bin/env python
"""
A Brown truss with a variable number of bays.

Created: 2025/10/31 18:35:08
Last modified: 2025/11/08 10:45:16
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import os
import time

import numpy as np

import pyfem


def brown_truss(bays: int) -> tuple[list[list[int]], np.ndarray]:
    """Define the mesh (nodal coordinates and connectivity table) for a
    two-dimensional Brown truss with any number of bays."""

    bay = np.array([[0, 1], [0, 2]])
    bay_tmp = np.array([[1, 2], [1, 3], [2, 4], [1, 4], [2, 3]])
    bay = np.vstack((bay, bay_tmp))

    for _ in range(bays - 1):
        bay_tmp = bay_tmp + 2
        bay = np.vstack((bay, bay_tmp))

    bay_last = np.zeros((3, 2))
    bay_last[0, :] = bay_tmp[4, :] + 1
    bay_last[1, :] = (bay_tmp[4, 1], bay_tmp[4, 1] + 2)
    bay_last[2, :] = (bay_tmp[4, 1] + 1, bay_tmp[4, 1] + 2)

    connectivity = np.vstack((bay, bay_last)).astype(int)

    x_coords = np.vstack(
        (
            np.array([[0]]),
            np.repeat(np.arange(1, bays + 2), 2).reshape(-1, 1),
            np.array([[bays + 2]]),
        )
    )
    y_coords = np.vstack(
        (np.array([[0]]), np.tile([0, 1], bays + 1).reshape(-1, 1), np.array([[0]]))
    )
    points = np.hstack((x_coords, y_coords))

    return connectivity.tolist(), points


def main(
    bays: int = 3, use_sparse: bool = True
) -> tuple[float, int, float, int, float]:
    # PREPROCESSING

    # 1. Geometry and discretization

    element_connectivity, points = brown_truss(bays)
    num_nodes = len(points)
    num_elements = len(element_connectivity)

    # 2. Element properties

    # Define element properties registry
    element_properties = pyfem.make_element_properties(
        [
            ("bar", ("bar_2D", {"E": 206000.0, "A": 500.0})),
        ]
    )

    # Assign properties to elements
    element_property_labels = ["bar"] * len(element_connectivity)

    # 3. Mesh

    # Create mesh
    mesh = pyfem.Mesh(
        num_nodes=num_nodes,
        points=points,
        num_elements=num_elements,
        element_connectivity=element_connectivity,
        element_property_labels=element_property_labels,
    )

    # 4. Create Model

    problem = pyfem.Problem(
        pyfem.Physics.MECHANICS,
        pyfem.Dimension.D2,
    )

    model = pyfem.Model(mesh, problem)
    model.set_element_properties(element_properties)
    print(model)

    # 5. Boundary conditions

    # Dirichlet boundary conditions (prescribed displacements)
    model.bc.prescribe_displacement(0, pyfem.DOFType.U_X, 0.0)
    model.bc.prescribe_displacement(0, pyfem.DOFType.U_Y, 0.0)
    model.bc.prescribe_displacement(num_nodes - 1, pyfem.DOFType.U_Y, 0.0)

    # Neumann boundary conditions (applied forces)
    model.bc.apply_force(1, pyfem.DOFType.U_Y, -10000.0)

    # print(f"\n- Prescribed displacements: {model.bc.prescribed_displacements}")
    # print(f"- Applied forces: {model.bc.applied_forces}")

    # PROCESSING: Solve FEA problem

    # Create solver
    solver = pyfem.LinearStaticSolver(model, use_sparse=use_sparse)

    # Assemble the global stiffness matrix
    solver.assemble_global_matrix()

    # Apply boundary conditions
    solver.apply_boundary_conditions()

    # Solve for nodal displacements
    solver.solve()

    # POSTPROCESSING: Compute derived quantities

    # Create postprocessor
    postprocessor = pyfem.PostProcessor(
        model.mesh,
        model.element_properties,
        solver.global_stiffness_matrix,
        solver.nodal_displacements,
        magnification_factor=1000.0,
    )

    # Plot truss
    postprocessor.undeformed_mesh()
    postprocessor.deformed_mesh()

    return (
        float(solver.nodal_displacements[num_nodes - 1]),
        solver.dof_space.total_dofs,
        solver.sparsity_percentage,
        solver.matrix_size_bytes,
        solver.solve_time,
    )


# ---------------------- examples


def benchmark_comparison():
    """Benchmark: Compare sparse vs dense solvers"""
    print("\n" + "=" * 70)
    print("BENCHMARK: Sparse vs Dense Solver Comparison")
    print("=" * 70)

    bay_sizes = [20, 200, 2000, 3000, 4000]

    results = []

    # Set the environment variable to prevent plot display
    os.environ["Show_TrussPlot"] = "0"

    for bays in bay_sizes:
        print(f"\n\n{'=' * 70}")
        print(f"Testing {bays} bays...")
        print(f"{'=' * 70}")

        # Dense solver
        print("\n--- DENSE SOLVER ---")
        start = time.time()
        _, dofs_d, sparsity_percentage_d, matrix_size_bytes_d, solve_time_d = main(
            bays, use_sparse=False
        )
        total_time_d = time.time() - start

        results.append(
            {
                "bays": bays,
                "solver": "Dense",
                "dofs": dofs_d,
                "sparsity": sparsity_percentage_d,
                "memory_mib": matrix_size_bytes_d / (1024 * 1024),
                "solve_time": solve_time_d,
                "total_time": total_time_d,
            }
        )

        # Sparse solver
        print("\n--- SPARSE SOLVER ---")
        start_time = time.time()
        _, dofs_s, sparsity_percentage_s, matrix_size_bytes_s, solve_time_s = main(
            bays, use_sparse=True
        )
        total_time_s = time.time() - start_time

        results.append(
            {
                "bays": bays,
                "solver": "Sparse",
                "dofs": dofs_s,
                "sparsity": sparsity_percentage_s,
                "memory_mib": matrix_size_bytes_s / (1024 * 1024),
                "solve_time": solve_time_s,
                "total_time": total_time_s,
            }
        )

    # Print summary table
    print("\n\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(
        f"{'Bays':<8} {'Solver':<10} {'DOFs':<10} {'Sparsity %':<12} {'Memory (MiB)':<14} {'Solve time (s)':<20} {'Total time (s)':<14}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r['bays']:<8} {r['solver']:<10} {r['dofs']:<10} {r['sparsity']:<12.2f} {r['memory_mib']:<14.3f} {r['solve_time']:<20.4f} {r['total_time']:<14.4f}"
        )

    # Calculate speedups
    print("\n" + "=" * 100)
    print("SPEEDUP ANALYSIS (Sparse vs Dense)")
    print("=" * 100)
    print(
        f"{'Bays':<8} {'Memory Reduction':<20} {'Solve Speedup':<20} {'Total Speedup':<20}"
    )
    print("-" * 100)

    for bays in bay_sizes:
        dense = next(
            (r for r in results if r["bays"] == bays and r["solver"] == "Dense"), None
        )
        sparse_r = next(
            (r for r in results if r["bays"] == bays and r["solver"] == "Sparse"), None
        )

        if dense and sparse_r:
            mem_reduction = dense["memory_mib"] / sparse_r["memory_mib"]
            solve_speedup = dense["solve_time"] / sparse_r["solve_time"]
            total_speedup = dense["total_time"] / sparse_r["total_time"]

            print(
                f"{bays:<8}"
                f"{(str(f'{mem_reduction:.2f}') + 'x'):<20}"
                f"{(str(f'{solve_speedup:.2f}') + 'x'):<20}"
                f"{(str(f'{total_speedup:.2f}') + 'x'):<20}"
            )

    print("=" * 100)

    # Reset the environment variable
    del os.environ["Show_TrussPlot"]


if __name__ == "__main__":
    main(bays=20, use_sparse=False)
    # main(bays=20, use_sparse=True)
    # benchmark_comparison()
