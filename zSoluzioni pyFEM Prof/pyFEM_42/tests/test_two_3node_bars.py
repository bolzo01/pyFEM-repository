"""
Test displacement solution for Two 3-node bars in series with varying cross sections, under tension.

Created: 2025/11/11 22:43:18
Last modified: 2025/11/11 22:44:02
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from examples.two_3node_bars import main


def test_two_3node_bars_output():
    expected_displacement = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        main(),
        expected_displacement,
        atol=1e-14,
        rtol=1e-14,
    )
