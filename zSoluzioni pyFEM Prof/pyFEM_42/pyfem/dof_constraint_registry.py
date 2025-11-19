#!/usr/bin/env python
"""
Registry for low-level DOF-based boundary conditions.

Created: 2024/11/06 17:32:52
Last modified: 2025/11/08 19:36:24
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from enum import Enum, auto


class ConstraintType(Enum):
    DIRICHLET = auto()
    NEUMANN = auto()


class DOFConstraintRegistry:
    """
    Central registry for all DOF constraints.

    Stores boundary conditions at the *global DOF* level.
    Tracks which global DOFs have been assigned what kind of constraint,
    and forbids assigning incompatible or conflicting constraints.
    """

    def __init__(self):
        # DOF index -> ConstraintType
        self._constraint_type = {}

        # DOF index -> prescribed displacement
        self._values = {}

        # DOF index -> accumulated Neumann force
        self._forces = {}

    def assign(self, dof: int, ctype: ConstraintType):
        """Register that this DOF has a constraint of the given type."""
        if dof in self._constraint_type:
            existing = self._constraint_type[dof]
            if existing != ctype:
                raise ValueError(
                    f"Cannot assign {ctype.name} to DOF {dof}: "
                    f"it already has {existing.name}"
                )
        else:
            self._constraint_type[dof] = ctype

    def set_dirichlet_value(self, dof: int, value: float):
        """Assign a Dirichlet value, preventing conflicts."""
        self.assign(dof, ConstraintType.DIRICHLET)

        if dof in self._values and self._values[dof] != value:
            raise ValueError(
                f"Dirichlet DOF {dof} already has value {self._values[dof]}, "
                f"cannot assign different value {value}"
            )

        self._values[dof] = value

    def add_neumann_force(self, dof: int, force: float):
        """Assign or accumulate a nodal force."""
        self.assign(dof, ConstraintType.NEUMANN)

        # Neumann forces are additive
        self._forces[dof] = self._forces.get(dof, 0.0) + force

    def get_dirichlet_values(self):
        """Returns dict[dof] = prescribed displacement."""
        return self._values

    def get_neumann_forces(self):
        """Returns dict[dof] = total applied nodal force."""
        return self._forces

    def get_constraint(self, dof: int):
        return self._constraint_type.get(dof, None)

    def has_constraint(self, dof: int) -> bool:
        return dof in self._constraint_type
