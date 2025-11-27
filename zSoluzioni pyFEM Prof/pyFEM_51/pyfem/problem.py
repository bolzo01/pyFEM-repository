#!/usr/bin/env python
"""
Module defining problem types and their DOF requirements.

Created: 2025/10/25 02:44:07
Last modified: 2025/10/29 00:07:09
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from dataclasses import dataclass
from enum import Enum, auto

from .dof_registry import get_dof_types_for_problem


class Physics(Enum):
    """Physical phenomena being modeled"""

    MECHANICS = auto()  # DOFs: ux, uy, uz
    HEAT_TRANSFER = auto()  # DOFs: T (temperature)
    THERMO_MECHANICS = auto()  # DOFs: ux, uy, uz, T (coupled)


class Dimension(Enum):
    D1 = 1
    D2 = 2
    D3 = 3


@dataclass(frozen=True)
class Problem:
    physics: Physics
    dimension: Dimension

    def __str__(self) -> str:
        """Return a concise, human-readable description of the problem."""
        physics_str = self.physics.name.replace("_", " ").title()

        # Get DOF types and format them
        try:
            dof_types = self.get_dof_types()
            dof_str = ", ".join(dt.value for dt in dof_types)
        except RuntimeError:
            dof_str = "undefined"

        return f"{physics_str} ({self.dimension.value}D) - DOFs: [{dof_str}]"

    def get_dof_types(self):
        """
        Return the list of DOF types associated with this problem.

        This method delegates the lookup to the DOF registry, using a key
        derived from the physics, and dimension of the problem.
        For example, a 1D mechanics problem corresponds to the key
        'mechanics_1d'.

        Raises:
            RuntimeError: If no DOF mapping is defined for this problem configuration.
        """
        key = f"{self.physics.name.lower()}_{self.dimension.value}d"

        try:
            return get_dof_types_for_problem(key)
        except KeyError:
            raise RuntimeError(
                f"No DOF mapping defined for problem: {self.physics.name} "
                f"({self.dimension.value}D"
            ) from None
