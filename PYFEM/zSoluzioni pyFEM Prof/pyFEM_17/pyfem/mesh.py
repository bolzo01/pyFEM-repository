#!/usr/bin/env python
"""
Module defining the Mesh class for finite element analysis.

Created: 2024/10/13 19:05:39
Last modified: 2025/10/18 22:33:36
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np


class Mesh:
    """Class representing the mesh data structure.

    This class stores the parameters needed to discretize a domain into
    a mesh for finite element analysis.
    """

    def __init__(
        self,
        num_nodes: int,
        points: np.ndarray,
        num_elements: int,
        element_connectivity: list[list[int]],
        element_material: list[str],
    ):
        # number of nodes
        self.num_nodes = num_nodes

        # number of points
        self.points = points

        # number of elements
        self.num_elements = num_elements

        # element connectivity matrix
        self.element_connectivity = element_connectivity

        # per-element material labels (one label per element)
        self.element_material = element_material
