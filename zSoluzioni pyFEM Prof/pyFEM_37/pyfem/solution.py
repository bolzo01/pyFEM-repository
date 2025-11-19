#!/usr/bin/env python
"""
Module defining the Solution class for FEA results.

Created: 2025/11/13 23:18:15
Last modified: 2025/11/14 01:28:44
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Solution:
    """Container for finite element analysis solution results.

    Attributes:
        nodal_displacements: Displacement vector for all DOFs
        solver_stats: Optional dictionary of solver performance metrics
    """

    nodal_displacements: np.ndarray
    solver_stats: dict[str, int | float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate solution data."""
        if not isinstance(self.nodal_displacements, np.ndarray):
            raise TypeError("nodal_displacements must be a numpy array")
        if self.nodal_displacements.ndim != 1:
            raise ValueError("nodal_displacements must be a 1D array")

    @property
    def num_dofs(self) -> int:
        """Total number of degrees of freedom in solution."""
        return len(self.nodal_displacements)
