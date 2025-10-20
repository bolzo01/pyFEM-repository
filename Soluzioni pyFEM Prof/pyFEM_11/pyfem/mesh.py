#!/usr/bin/env python
"""
Module defining the Mesh class for finite element analysis.

Created: 2024/10/13 19:05:39
Last modified: 2025/10/12 22:23:36
Author: Angelo Simone (angelo.simone@unipd.it)
"""


class Mesh:
    """Class representing the mesh data structure.

    This class stores the parameters needed to discretize a domain into
    a mesh for finite element analysis.
    """

    def __init__(
        self, num_nodes: int, num_elements: int, element_connectivity: list[list[int]]
    ):
        # number of nodes
        self.num_nodes = num_nodes

        # number of elements
        self.num_elements = num_elements

        # element connectivity matrix
        self.element_connectivity = element_connectivity
