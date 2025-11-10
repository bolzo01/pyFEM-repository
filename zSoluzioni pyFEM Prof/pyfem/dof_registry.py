#!/usr/bin/env python
"""
Registry mapping problem identifiers to their required DOF types.

This module defines the mapping using string keys that correspond
to the `Problem` enum values.

Created: 2025/10/26 01:39:28
Last modified: 2025/11/01 15:43:42
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from .dof_types import DOFType

DOF_REQUIREMENTS: dict[str, list[DOFType]] = {
    "mechanics_1d": [
        DOFType.U_X,
    ],
    "heat_transfer_1d": [
        DOFType.TEMPERATURE,
    ],
    "heat_transfer_2d": [
        DOFType.TEMPERATURE,
    ],
    "mechanics_2d": [
        DOFType.U_X,
        DOFType.U_Y,
    ],
    "mechanics_3d": [
        DOFType.U_X,
        DOFType.U_Y,
        DOFType.U_Z,
    ],
    "thermo_mechanics_1d": [
        DOFType.U_X,
        DOFType.TEMPERATURE,
    ],
}


def get_dof_types_for_problem(problem_identifier: str) -> list[DOFType]:
    """Return DOF types for the given problem identifier (string).

    Args:
        problem: Problem type

    Returns:
        List of required DOF types

    Example:
        dof_types = get_dof_types_for_problem(Problem.MECHANICS_3D)
        # Returns: [DOFType.U_X, DOFType.U_Y, DOFType.U_Z]
    """

    try:
        return DOF_REQUIREMENTS[problem_identifier]
    except KeyError:
        raise KeyError(f"Unknown problem identifier: {problem_identifier}")
