"""
Test displacement solution for bar in tension (in 2D).

Created: 2025/10/31 15:00:03
Last modified: 2025/10/31 15:30:14
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from examples.one_bar_in_2d import main


def test_one_bar_output():
    expected_displacement = np.array([0.0, 0.0, 12.0 * 10.0 / 23.2 / 7.0, 0.0])
    np.testing.assert_allclose(
        main(),
        expected_displacement,
        atol=1e-14,
        rtol=1e-14,
    )
