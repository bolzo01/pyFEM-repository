"""
Framework for defining finite element formulations.

This module implements the FiniteElement abstract base class,
encapsulating the common operations needed in finite element analysis.

Created: 2025/11/16 21:18:34
Last modified: 2025/11/17 00:56:15
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from abc import ABC, abstractmethod

import numpy as np


class FiniteElement(ABC):
    """
    Abstract base class for all finite element formulations.

    This class defines the core interface that every finite element in the
    library must implement. A FiniteElement encapsulates:

    - geometric interpolation (shape functions, derivatives, Jacobian)
    - numerical integration (Gauss points)
    - the local DOF structure of the element
    - element-level operators (B-matrix, stiffness matrix)

    Responsibilities
    ----------------
    Geometry and interpolation
        - evaluate shape functions and their derivatives
        - compute the Jacobian and mapping from reference to physical space
        - provide integration points and weights

    Element operators
        - construct the strain-displacement matrix (B-matrix)
        - assemble linear stiffness matrices

    Notes
    -----
    All element types (e.g., bars, beams, triangles, quads, bricks)
    must derive from this class and implement the interpolation and
    integration routines appropriate to their geometry. Subclasses may also
    override the default B-matrix or stiffness implementations when closed-form
    expressions are available.
    """

    # Basic properties

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """Number of nodes of the element."""
        pass

    @property
    @abstractmethod
    def dofs_per_node(self) -> int:
        """Degrees of freedom per node (e.g., 1, 2, or 3)."""
        pass

    @property
    def num_dofs(self) -> int:
        return self.num_nodes * self.dofs_per_node

    # Geometry and interpolation

    @abstractmethod
    def shape_functions(self, xi) -> np.ndarray:
        """N(xi): shape functions evaluated at reference coordinate xi."""
        pass

    @abstractmethod
    def shape_function_derivatives(self, xi) -> np.ndarray:
        """dN/dxi at reference coordinate xi."""
        pass

    @abstractmethod
    def integration_points(self):
        """
        Return list of (xi, weight) tuples for numerical integration.
        """
        pass

    @abstractmethod
    def jacobian(self, x_nodes, xi) -> float:
        """Jacobian determinant |d x / d xi|."""
        pass

    def B_matrix(self, x_nodes, xi):
        """
        Default B-matrix computation for isoparametric elements:
        Converts dN/dxi to dN/dx.

        Can be overridden by closed-form elements.
        """
        dN_dxi = self.shape_function_derivatives(xi)  # shape: (num_nodes,)
        J = self.jacobian(x_nodes, xi)

        if J <= 0.0:
            raise ValueError(f"Invalid Jacobian: {J}")

        dN_dx = dN_dxi / J  # shape: (num_nodes,)

        # Build strain-displacement matrix: element-dependent
        # Default = 1D axial strain
        return dN_dx.reshape(1, self.num_nodes)

    # Element-level computations

    def compute_stiffness(self, material, x_nodes):
        """
        Default stiffness matrix using numerical integration.
        Override for closed-form elements.
        """
        K = np.zeros((self.num_dofs, self.num_dofs))
        D = material.constitutive_matrix

        for xi, w in self.integration_points():
            B = self.B_matrix(x_nodes, xi)
            J = self.jacobian(x_nodes, xi)
            K += B.T @ D @ B * w * J

        return K
