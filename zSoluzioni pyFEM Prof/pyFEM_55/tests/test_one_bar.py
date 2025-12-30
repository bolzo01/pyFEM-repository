"""
Test displacement solution for bar in tension.

Created: 2025/10/30 18:52:42
Last modified: 2025/10/30 19:03:12
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from examples.one_bar import main


def test_one_bar_output():
    expected_displacement = np.array([0.0, 12.0 * 10.0 / 23.2 / 7.0])
    np.testing.assert_allclose(
        main(),
        expected_displacement,
        atol=1e-14,
        rtol=1e-14,
    )
