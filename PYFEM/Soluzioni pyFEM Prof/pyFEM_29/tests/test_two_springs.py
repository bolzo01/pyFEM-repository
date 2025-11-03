"""
Test displacement solution for two springs in series.

Created: 2025/10/19 13:25:14
Last modified: 2025/10/19 17:39:24
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from examples.two_springs import main


def test_two_springs_output():
    expected_displacement = np.array([15.0, 0.0, 10.0])
    np.testing.assert_allclose(
        main(),
        expected_displacement,
        atol=1e-14,
        rtol=1e-14,
    )
