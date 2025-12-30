"""
Test displacement solution for Two-bar truss - Example 2 in Trusses.pdf.

Created: 2025/10/31 15:00:03
Last modified: 2025/11/01 15:48:42
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from examples.two_bar_truss import main


def test_two_bar_truss_output():
    expected_displacement = np.array([0.0, 0.0, 1 / 6, 1 / 3, 0.0, 0.0])
    np.testing.assert_allclose(
        main(),
        expected_displacement,
        atol=1e-14,
        rtol=1e-14,
    )
