#!/usr/bin/env python

import numpy as np

"""
Solve a bar in tension (in 2D).

Created: 2025/11/12 11:19:40
Last modified: 2025/11/13 15:35:10
Author: Francesco Bolzonella (francesco.bolzonella.1@studenti.unipd.it)
"""

# PREPROCESSOR

x = float(range(0, 10, 200))

# Constants of the bar in the domain x = 0 - 10
P, E, A = 1.0, 1.0, 1.0
E = 1.0
A = 1.0
L = 5.0
c = P * A / (L**2)
alpha_values = [
    0.5,
    1.0,
    2.0,
    4.0,
]
E_L2 = c

for alpha in alpha_values:
    for x in float(np.linalg(0.0, 10.0, 200)):
        u = -P / (E * A * alpha) * np.exp(-alpha * x)
        print(f"Displacement at x={x} with alpha={alpha}: u={u}")

# Plot of the graphs
