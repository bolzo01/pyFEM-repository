#!/usr/bin/env python
"""
PyTest patch tests for the 4-node linear tetrahedral element.

Tests:
1) Hydrostatic compression
2) Pure shear gamma_xy
3) Uniaxial tension in x-direction

Created: 2025/11/23 19:04:04
Last modified: 2025/11/23 19:11:10
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


# -------------------------------------------------------------
# Small helper to compute stress in one tetra element
# -------------------------------------------------------------
def compute_stress(x_nodes, u_nodes, E, nu):
    mat = pyfem.LinearElastic3D(E=E, nu=nu)

    # Create element manually and assign an index
    from pyfem.elements.tetra import Tetra

    elem = Tetra({}, {})
    elem.element_index = 0

    return elem.compute_stress(mat, x_nodes, u_nodes)


# -------------------------------------------------------------
# Test 1 — Hydrostatic compression
# -------------------------------------------------------------
def test_tetra_hydrostatic():
    E, nu = 200000.0, 0.3
    eps = -0.01  # compressive
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)

    u = eps * x  # Uniform volumetric strain
    u = u.reshape(-1)

    sigma = compute_stress(x, u, E, nu)

    # Analytical 3D hydrostatic stress
    K = E / (3 * (1 - 2 * nu))
    p = 3 * K * eps
    sigma_exact = np.array([p, p, p, 0, 0, 0])

    np.testing.assert_allclose(sigma, sigma_exact, rtol=1e-10, atol=1e-10)


# -------------------------------------------------------------
# Test 2 — Pure shear gamma_xy
# -------------------------------------------------------------
def test_tetra_shear_xy():
    E, nu = 200000.0, 0.3
    gamma = 0.02

    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    u = np.zeros_like(x)
    u[:, 0] = gamma * x[:, 1]  # u_x = gamma_xy * y
    u = u.reshape(-1)

    sigma = compute_stress(x, u, E, nu)

    # Analytical tau_xy = G gamma_xy
    G = E / (2 * (1 + nu))
    tau = G * gamma
    sigma_exact = np.array([0, 0, 0, tau, 0, 0])

    np.testing.assert_allclose(sigma, sigma_exact, rtol=1e-10, atol=1e-10)


# -------------------------------------------------------------
# Test 3 — Uniaxial tension in x-direction
# -------------------------------------------------------------
def test_tetra_uniaxial_x():
    E, nu = 200000.0, 0.3
    eps = 0.01

    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    u = np.zeros_like(x)
    u[:, 0] = eps * x[:, 0]  # u_x = eps*x
    u = u.reshape(-1)

    sigma = compute_stress(x, u, E, nu)

    # Correct analytical stress for epsilon=[eps_xx,0,0,0,0,0]
    coeff = E / ((1 + nu) * (1 - 2 * nu))
    sigma_x = coeff * (1 - nu) * eps
    sigma_y = coeff * nu * eps
    sigma_z = coeff * nu * eps
    sigma_exact = np.array([sigma_x, sigma_y, sigma_z, 0, 0, 0])

    np.testing.assert_allclose(sigma, sigma_exact, rtol=1e-10, atol=1e-10)
