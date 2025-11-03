#!/usr/bin/env python
"""
Module defining the PostProcessor class.

Created: 2025/11/02 13:07:50
Last modified: 2025/11/03 15:26:12
Author: Francesco Bolzonella (francesco.bolzonella.1@studenti.unipd.it)
"""

import numpy as np

from .element_properties import ElementProperties, param
from .mesh import Mesh


class PostProcessor:
    """Handles post-processing computations for finite element analysis.

    Computes strain energy and other derived quantities from FEA solutions.
    """

    def __init__(
        self,
        mesh: Mesh,
        element_properties: ElementProperties,
        original_global_stiffness_matrix: np.ndarray,
        nodal_displacements: np.ndarray,
    ):
        self.mesh = mesh
        self.element_properties = element_properties
        self.original_global_stiffness_matrix = original_global_stiffness_matrix
        self.nodal_displacements = nodal_displacements

    def compute_strain_energy_local(self) -> None:
        """
        Computes the total strain energy by summing the strain energy of each element.

        Returns:
            None.
        """

        num_elements = self.mesh.num_elements
        element_connectivity = self.mesh.element_connectivity

        total_strain_energy = 0.0
        for element_index in range(num_elements):
            label = self.mesh.element_property_labels[element_index]
            elem_prop = self.element_properties[label]
            k_spring = float(param(elem_prop, "k", float))
            node1, node2 = element_connectivity[element_index]
            u1 = self.nodal_displacements[node1]
            u2 = self.nodal_displacements[node2]
            delta = u2 - u1
            strain_energy = 0.5 * k_spring * delta**2
            total_strain_energy += strain_energy
            print(f"\n- Strain energy in element {element_index}: {strain_energy}")
        print(
            f"\n- Total strain energy in the system (from local computation): {total_strain_energy}"
        )

    def compute_strain_energy_global(self) -> None:
        """
        Computes the total strain energy using the global solution (U = 0.5 * u^T * K * u).

        Returns:
            None.
        """

        K = self.original_global_stiffness_matrix
        u = self.nodal_displacements
        total_strain_energy = 0.5 * (u.T @ (K @ u))

        print(
            f"\n- Total strain energy in the system (from global computation): {total_strain_energy}"
        )
