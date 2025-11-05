#!/usr/bin/env python
"""
Module defining the Mesh class for finite element analysis.

Created: 2024/10/13 19:05:39
Last modified: 2025/10/12 21:14:11
Author: Angelo Simone (angelo.simone@unipd.it)
"""


class Mesh:
    """Class representing the mesh data structure.

    This class stores the parameters needed to discretize a domain into
    a mesh for finite element analysis.
    """

    # number of nodes
    num_nodes: int

    # number of elements
    num_elements: int

    # element connectivity matrix
    element_connectivity: list[list[int]]
