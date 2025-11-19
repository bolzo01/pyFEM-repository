#!/usr/bin/env python
"""
Quadrature rules for numerical integration in finite element analysis.

Created: 2025/11/08 23:11:28
Last modified: 2025/11/11 22:57:37
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from typing import Protocol

import numpy as np


class QuadratureRule(Protocol):
    """Protocol for quadrature rules."""

    @property
    def points(self) -> np.ndarray:
        """Quadrature points in reference coordinates."""
        ...

    @property
    def weights(self) -> np.ndarray:
        """Quadrature weights."""
        ...

    @property
    def n_points(self) -> int:
        """Number of quadrature points."""
        ...


class GaussLegendre1D:
    """Gauss-Legendre quadrature for 1D line elements on [-1, 1].

    This is the optimal quadrature rule for polynomials on the interval [-1, 1].
    An n-point rule integrates polynomials of degree ≤ 2n-1 exactly.

    Args:
        order: Number of quadrature points (1, 2, 3, ...)

    Examples:
        >>> quad = GaussLegendre1D(order=2)
        >>> quad.points
        array([-0.57735027,  0.57735027])
        >>> quad.weights
        array([1., 1.])
        >>> quad.n_points
        2
    """

    def __init__(self, order: int):
        if order < 1:
            raise ValueError(f"Order must be >= 1, got {order}")

        self._order = order
        # Get Gauss-Legendre points and weights using NumPy
        self._points, self._weights = np.polynomial.legendre.leggauss(order)

    @property
    def points(self) -> np.ndarray:
        """Quadrature points in reference element [-1, 1], shape (n_points,)."""
        return self._points

    @property
    def weights(self) -> np.ndarray:
        """Quadrature weights, shape (n_points,)."""
        return self._weights

    @property
    def n_points(self) -> int:
        """Number of quadrature points."""
        return self._order

    def __repr__(self) -> str:
        return f"GaussLegendre1D(order={self._order})"


class GaussLegendre2D:
    """Gauss-Legendre quadrature for 2D quad elements on [-1, 1] x [-1, 1].

    Uses tensor product of 1D Gauss-Legendre rules.

    Args:
        order: Number of quadrature points per direction

    Examples:
        >>> quad = GaussLegendre2D(order=2)
        >>> quad.n_points
        4
        >>> quad.points.shape
        (4, 2)
    """

    def __init__(self, order: int):
        if order < 1:
            raise ValueError(f"Order must be >= 1, got {order}")

        self._order = order
        # Get 1D points and weights
        xi_1d, w_1d = np.polynomial.legendre.leggauss(order)

        # Create tensor product grid
        xi, eta = np.meshgrid(xi_1d, xi_1d, indexing="ij")
        self._points = np.column_stack([xi.ravel(), eta.ravel()])

        # Tensor product weights
        w_xi, w_eta = np.meshgrid(w_1d, w_1d, indexing="ij")
        self._weights = (w_xi * w_eta).ravel()

    @property
    def points(self) -> np.ndarray:
        """Quadrature points in reference element, shape (n_points, 2)."""
        return self._points

    @property
    def weights(self) -> np.ndarray:
        """Quadrature weights, shape (n_points,)."""
        return self._weights

    @property
    def n_points(self) -> int:
        """Number of quadrature points."""
        return self._order**2

    def __repr__(self) -> str:
        return f"GaussLegendre2D(order={self._order})"


class TriangleQuadrature:
    """Symmetric quadrature rules for triangular elements.

    Standard triangle with vertices at (0,0), (1,0), (0,1).

    Args:
        order: Quadrature order (1, 2, or 3)
            - order=1: 1 point (centroid), exact for linear polynomials
            - order=2: 3 points (midpoints), exact for quadratic polynomials
            - order=3: 4 points, exact for cubic polynomials
    """

    # Pre-computed symmetric quadrature rules for triangles
    _RULES = {
        1: {  # 1-point rule (centroid)
            "points": np.array([[1 / 3, 1 / 3]]),
            "weights": np.array([0.5]),
        },
        2: {  # 3-point rule
            "points": np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]),
            "weights": np.array([1 / 6, 1 / 6, 1 / 6]),
        },
        3: {  # 4-point rule
            "points": np.array(
                [[1 / 3, 1 / 3], [3 / 5, 1 / 5], [1 / 5, 3 / 5], [1 / 5, 1 / 5]]
            ),
            "weights": np.array([-27 / 96, 25 / 96, 25 / 96, 25 / 96]),
        },
    }

    def __init__(self, order: int):
        if order not in self._RULES:
            raise ValueError(
                f"Order must be 1, 2, or 3 for triangles, got {order}. "
                f"Available orders: {sorted(self._RULES.keys())}"
            )

        self._order = order
        rule = self._RULES[order]
        self._points = rule["points"]
        self._weights = rule["weights"]

    @property
    def points(self) -> np.ndarray:
        """Quadrature points in reference triangle, shape (n_points, 2)."""
        return self._points

    @property
    def weights(self) -> np.ndarray:
        """Quadrature weights, shape (n_points,)."""
        return self._weights

    @property
    def n_points(self) -> int:
        """Number of quadrature points."""
        return len(self._weights)

    def __repr__(self) -> str:
        return f"TriangleQuadrature(order={self._order})"


def get_quadrature_rule(
    element_kind: str, integration_scheme: str | int
) -> QuadratureRule | None:
    """Factory function to create quadrature rules based on element type.

    Args:
        element_kind: Element type ('bar_1D', 'bar3_1D', 'quad4', 'tri3', etc.)
        integration_scheme: Integration scheme specification
            - 'analytical': Return None (use analytical integration)
            - 'full': Full integration
            - 'reduced': Reduced integration
            - Integer (1, 2, 3): Specific number of points

    Returns:
        QuadratureRule instance or None for analytical integration

    Examples:
        >>> rule = get_quadrature_rule('quad8', 'reduced')
        >>> rule.n_points
        4
        >>> rule = get_quadrature_rule('quad8', 3)
        >>> rule.n_points
        9
    """
    if integration_scheme == "analytical":
        return None

    # Determine default integration order based on element type
    if element_kind in ("bar_1D"):
        # Linear 2-node bar: 1 Gauss point is exact
        if integration_scheme == "full":
            order = 1
        elif integration_scheme == "reduced":
            order = 1  # same as full for linear bar
        else:
            try:
                order = int(integration_scheme)
            except ValueError:
                raise ValueError(
                    f"Invalid integration scheme '{integration_scheme}' for {element_kind}. "
                    f"Use 'full', 'reduced', or an integer."
                )
        return GaussLegendre1D(order)

    elif element_kind in ("bar3_1D"):
        # 3-node bar element: 2 Gauss point is exact
        if integration_scheme == "full":
            order = 2
        elif integration_scheme == "reduced":
            raise ValueError(
                f"Invalid integration scheme '{integration_scheme}' for {element_kind}. "
                f"Use 'full' or an integer."
            )
        else:
            try:
                order = int(integration_scheme)
            except ValueError:
                raise ValueError(
                    f"Invalid integration scheme '{integration_scheme}' for {element_kind}. "
                    f"Use 'full', 'reduced', or an integer."
                )
        return GaussLegendre1D(order)

    elif element_kind == "quad4":
        # Bilinear quadrilateral
        if integration_scheme == "full":
            order = 2  # 2x2
        elif integration_scheme == "reduced":
            order = 1  # 1x1
        else:
            try:
                order = int(integration_scheme)
            except ValueError:
                raise ValueError(
                    f"Invalid integration scheme '{integration_scheme}' for quad4."
                )
        return GaussLegendre2D(order)

    elif element_kind == "quad8":
        # Quadratic serendipity quad
        if integration_scheme == "full":
            order = 3  # 3x3 recommended for quadratic serendipity
        elif integration_scheme == "reduced":
            order = 2  # 2x2 reduced (still accurate)
        else:
            try:
                order = int(integration_scheme)
            except ValueError:
                raise ValueError(
                    f"Invalid integration scheme '{integration_scheme}' for quad8."
                )
        return GaussLegendre2D(order)

    elif element_kind == "tri3":
        # Linear triangle (CST)
        if integration_scheme == "full":
            order = 1  # 1-point is the exact CST rule
        elif integration_scheme == "reduced":
            order = 1  # same – CST has constant strain
        else:
            try:
                order = int(integration_scheme)
            except ValueError:
                raise ValueError(
                    f"Invalid integration scheme '{integration_scheme}' for tri3."
                )
        return TriangleQuadrature(order)

    elif element_kind == "tri6":
        # Quadratic triangle (LST)
        if integration_scheme == "full":
            order = 3  # standard 3-point quadrature
        elif integration_scheme == "reduced":
            order = 2  # 2-point reduced rule
        else:
            try:
                order = int(integration_scheme)
            except ValueError:
                raise ValueError(
                    f"Invalid integration scheme '{integration_scheme}' for tri6."
                )
        return TriangleQuadrature(order)

    else:
        raise NotImplementedError(
            f"Quadrature rules for element kind '{element_kind}' not implemented"
        )
