#!/usr/bin/env python
"""
Element-problem compatibility registry and validation.

Created: 2025/10/30 01:11:17
Last modified: 2025/10/31 10:56:56
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from .problem import Dimension, Physics, Problem

# Registry mapping element kinds to compatible problem configurations
# Key: element kind, Value: set of (Physics, Dimension) tuples
ELEMENT_PROBLEM_COMPATIBILITY: dict[str, set[tuple[Physics, Dimension]]] = {
    # 1D Elements
    "spring_1D": {
        (Physics.MECHANICS, Dimension.D1),
    },
    "bar_1D": {
        (Physics.MECHANICS, Dimension.D1),
    },
    # 2D Mechanics Elements
    "bar_2D": {
        (Physics.MECHANICS, Dimension.D2),
    },
    "tri3": {
        (Physics.MECHANICS, Dimension.D2),
    },
    "tri6": {
        (Physics.MECHANICS, Dimension.D2),
    },
    "quad4": {
        (Physics.MECHANICS, Dimension.D2),
    },
    "quad8": {
        (Physics.MECHANICS, Dimension.D2),
    },
    # 2D Heat Transfer Elements
    "tri3_heat": {
        (Physics.HEAT_TRANSFER, Dimension.D2),
    },
    "quad4_heat": {
        (Physics.HEAT_TRANSFER, Dimension.D2),
    },
    # 3D Mechanics Elements
    "tet4": {
        (Physics.MECHANICS, Dimension.D3),
    },
    "tet10": {
        (Physics.MECHANICS, Dimension.D3),
    },
    "hex8": {
        (Physics.MECHANICS, Dimension.D3),
    },
    "hex20": {
        (Physics.MECHANICS, Dimension.D3),
    },
    # Multiphysics elements
    "quad4_thermo_mechanics": {
        (Physics.THERMO_MECHANICS, Dimension.D2),
    },
}


def is_element_compatible_with_problem(element_kind: str, problem: Problem) -> bool:
    """Check if an element kind is compatible with a problem.

    Args:
        element_kind: Element type identifier (e.g., "bar_1D", "quad4")
        problem: Problem instance

    Returns:
        True if compatible, False otherwise

    Example:
        problem = Problem(Physics.MECHANICS, Dimension.D1)
        is_compatible = is_element_compatible_with_problem("bar_1D", problem)
        # Returns: True

        is_compatible = is_element_compatible_with_problem("quad4", problem)
        # Returns: False
    """
    if element_kind not in ELEMENT_PROBLEM_COMPATIBILITY:
        # Unknown element kind
        return False

    # Check if (physics, dimension) tuple is in the compatible set
    compatible_configs = ELEMENT_PROBLEM_COMPATIBILITY[element_kind]
    return (problem.physics, problem.dimension) in compatible_configs


def validate_element_problem_compatibility(
    element_kinds: list[str], problem: Problem
) -> list[str]:
    """Validate that all element kinds are compatible with the problem.

    Args:
        element_kinds: List of element type identifiers used in the model
        problem: Problem instance

    Returns:
        List of incompatible element kinds (empty if all compatible)

    Example:
        problem = Problem(Physics.MECHANICS, Dimension.D1)
        element_kinds = ["bar_1D", "spring_1D"]
        incompatible = validate_element_problem_compatibility(element_kinds, problem)
        # Returns: [] (all compatible)

        element_kinds = ["bar_1D", "quad4"]
        incompatible = validate_element_problem_compatibility(element_kinds, problem)
        # Returns: ["quad4"] (quad4 is not compatible with 1D problems)
    """
    incompatible = []

    for element_kind in element_kinds:
        if not is_element_compatible_with_problem(element_kind, problem):
            incompatible.append(element_kind)

    return incompatible


def get_compatible_problems(element_kind: str) -> set[tuple[Physics, Dimension]]:
    """Get all problem configurations compatible with an element kind.

    Args:
        element_kind: Element type identifier

    Returns:
        Set of compatible (Physics, Dimension) tuples

    Raises:
        KeyError: If element kind not found in registry

    Example:
        configs = get_compatible_problems("quad4")
        # Returns: {(Physics.MECHANICS, Dimension.D2)}
    """
    if element_kind not in ELEMENT_PROBLEM_COMPATIBILITY:
        raise KeyError(
            f"Element kind '{element_kind}' not found in compatibility registry. "
            f"Known elements: {sorted(ELEMENT_PROBLEM_COMPATIBILITY.keys())}"
        )

    return ELEMENT_PROBLEM_COMPATIBILITY[element_kind]
