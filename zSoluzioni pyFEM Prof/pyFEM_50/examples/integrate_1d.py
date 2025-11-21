#!/usr/bin/env python
"""
Numerical integration examples.

Created: 2025/11/09 15:04:23
Last modified: 2025/11/16 13:22:20
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from typing import Protocol

import numpy as np

from pyfem.quadrature import GaussLegendre1D


# --- Function Protocol (vectorized version)
class NumericFunction(Protocol):
    """A function that maps an array of points to numerical values."""

    def __call__(self, x: np.ndarray) -> np.ndarray: ...


# --- Vectorized Gauss-Legendre integration
def integrate_1d_vectorized(
    func: NumericFunction,
    a: float,
    b: float,
    quad: GaussLegendre1D,
) -> float:
    """Integrate a function over [a, b] using Gauss-Legendre quadrature (vectorized)."""
    # Map reference points to physical interval
    xi = quad.points
    x = 0.5 * (b - a) * xi + 0.5 * (a + b)
    jacobian = 0.5 * (b - a)
    f_values = func(x)
    return jacobian * np.sum(quad.weights * f_values)


# --- Function Protocol (scalar version)
class ScalarFunction(Protocol):
    """A function that takes a scalar float and returns a scalar float."""

    def __call__(self, x: float) -> float: ...


# --- Loop-based Gauss-Legendre integration
def integrate_1d_loops(
    func: ScalarFunction,
    a: float,
    b: float,
    quadrature_rule: GaussLegendre1D,
) -> float:
    """Integrates a function over [a, b] using loop-based Gauss-Legendre quadrature."""
    xi = quadrature_rule.points
    weights = quadrature_rule.weights
    n = quadrature_rule.n_points
    jacobian = 0.5 * (b - a)

    integral = 0.0
    for ip in range(n):
        x_ip = map_to_interval(float(xi[ip]), a, b)
        integral += func(x_ip) * float(weights[ip]) * jacobian

    return integral


# --- Generic mapping function
def map_to_interval(xi: float, a: float, b: float) -> float:
    """Map xi from [-1,1] to [a,b]."""
    return 0.5 * (b - a) * xi + 0.5 * (a + b)


# --- Analytical solution for testing
def f_analytical(a: float, b: float) -> float:
    """Analytical integral of f(x) = x^2 + 1 over [a, b].

    int(x^2 + 1)dx = x^3/3 + x
    """
    return (b**3 / 3 + b) - (a**3 / 3 + a)


# --- Test functions
def f_loop(x: float) -> float:
    """Scalar version: f(x) = x^2 + 1"""
    return x**2 + 1


def f_vect(x: np.ndarray) -> np.ndarray:
    """Vectorized version: f(x) = x^2 + 1"""
    return x**2 + 1


def main() -> None:
    orders = [1, 2, 3]
    intervals = [(-1.0, 1.0), (0.0, 3.0), (-2.0, 2.0)]

    print("=" * 70)
    print("Gauss-Legendre Integration Test: f(x) = x^2 + 1")
    print("=" * 70)

    for a, b in intervals:
        print(f"\n{'-' * 70}")
        print(f"Integrating f(x) = x^2 + 1 over [{a}, {b}]")
        print(f"{'-' * 70}")

        for order in orders:
            quadrature_rule = GaussLegendre1D(order=order)

            # Vectorized version
            integral_vectorized = integrate_1d_vectorized(f_vect, a, b, quadrature_rule)

            # Loop-based version
            integral_loops = integrate_1d_loops(f_loop, a, b, quadrature_rule)

            # Analytical solution
            analytical = f_analytical(a, b)

            print(f"\n  Order {order} ({quadrature_rule.n_points} point(s)):")
            print(f"    Vectorized    = {integral_vectorized:.10f}")
            print(f"    Loop-based    = {integral_loops:.10f}")
            print(f"    Analytical    = {analytical:.10f}")
            print(f"    Error (Vec)   = {abs(integral_vectorized - analytical):.6e}")
            print(f"    Error (Loops) = {abs(integral_loops - analytical):.6e}")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()

    # Additional examples with lambda functions
    print("Additional Examples:")
    print("-" * 70)

    quad = GaussLegendre1D(order=3)

    result_lambda = integrate_1d_vectorized(lambda x: x**2 + 1, 0, 1, quad)
    result_func = integrate_1d_vectorized(f_vect, 0, 1, quad)
    analytical_01 = f_analytical(0, 1)

    print("\nIntegrating f(x) = x^2 + 1 over [0, 1]:")
    print(f"  Using lambda:     {result_lambda:.10f}")
    print(f"  Using function:   {result_func:.10f}")
    print(f"  Analytical:       {analytical_01:.10f}")
    print(f"  Error (lambda):   {abs(result_lambda - analytical_01):.6e}")
    print(f"  Error (function): {abs(result_func - analytical_01):.6e}")
