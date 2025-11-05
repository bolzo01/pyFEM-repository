"""
Test displacement solution for series combination of one spring and two bars in tension.

Created: 2025/10/19 13:37:45
Last modified: 2025/11/05 11:55:51
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

from examples.one_spring_two_bars import main


def test_two_springs_output():
    expected_displacement = np.array([0.0, 1.6, 2.4, 4.0])
    np.testing.assert_allclose(
        main(),
        expected_displacement,
        atol=1e-14,
        rtol=1e-14,
    )
