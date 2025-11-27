"""
Test displacement solution for Brown truss example.

Created: 2025/10/31 15:00:03
Last modified: 2025/11/14 01:31:10
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import os

import numpy as np

from examples.brown_truss import main


def test_brown_truss_output():
    # Set the environment variable to prevent plot display
    os.environ["Show_TrussPlot"] = "0"

    expected_displacement = -0.0019066054231800654
    solution = main(bays=10)
    dof = 23
    computed_displacement = solution.nodal_displacements[dof]

    np.testing.assert_allclose(
        computed_displacement,
        expected_displacement,
        atol=1e-14,
        rtol=1e-14,
    )

    # Reset the environment variable
    del os.environ["Show_TrussPlot"]


def test_brown_truss_sparse_vs_dense():
    # Set the environment variable to prevent plot display
    os.environ["Show_TrussPlot"] = "0"

    solution_s = main(bays=10, use_sparse=True)
    solution_d = main(bays=10, use_sparse=False)

    computed_displacement_s = solution_s.nodal_displacements
    computed_displacement_d = solution_d.nodal_displacements

    np.testing.assert_allclose(
        computed_displacement_s,
        computed_displacement_d,
        atol=1e-14,
        rtol=1e-14,
    )

    # Reset the environment variable
    del os.environ["Show_TrussPlot"]
