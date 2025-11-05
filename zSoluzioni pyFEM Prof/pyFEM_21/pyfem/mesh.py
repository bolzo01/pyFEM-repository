#!/usr/bin/env python
"""
Module defining the Mesh class for finite element analysis.

Created: 2024/10/13 19:05:39
Last modified: 2025/10/19 00:59:20
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np


class Mesh:
    """Represents the finite element mesh data structure.

    Stores discretization information including nodal coordinates, element
    connectivity, and element property assignments.
    """

    def __init__(
        self,
        num_nodes: int,
        points: np.ndarray,
        num_elements: int,
        element_connectivity: list[list[int]],
        element_property_labels: list[str],
    ):
        # number of nodes
        self.num_nodes = num_nodes

        # number of points
        self.points = points

        # number of elements
        self.num_elements = num_elements

        # element connectivity matrix
        self.element_connectivity = element_connectivity

        # per-element property labels (one label per element)
        self.element_property_labels = element_property_labels
