"""
Test displacement solution for Brown truss example.

Created: 2025/10/31 15:00:03
Last modified: 2025/11/01 16:03:08
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import os

import numpy as np

from examples.brown_truss import main


def test_brown_truss_output():
    # Set the environment variable to prevent plot display
    os.environ["Show_TrussPlot"] = "0"

    expected_displacement = -0.0019066054231800654
    computed_displacement, _, _, _ = main(bays=10)
    np.testing.assert_allclose(
        computed_displacement,
        expected_displacement,
        atol=1e-14,
        rtol=1e-14,
    )

    # Reset the environment variable
    del os.environ["Show_TrussPlot"]
