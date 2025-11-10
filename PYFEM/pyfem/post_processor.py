#!/usr/bin/env python
"""
Module defining the PostProcessor class.

Created: 2025/10/18 18:03:29
Last modified: 2025/11/06 22:16:17
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import os

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
        global_stiffness_matrix: np.ndarray,
        nodal_displacements: np.ndarray,
        magnification_factor: float = 0.0,
    ):
        self.mesh = mesh
        self.element_properties = element_properties
        self.global_stiffness_matrix = global_stiffness_matrix
        self.nodal_displacements = nodal_displacements
        self.magnification = magnification_factor

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

        K = self.global_stiffness_matrix
        u = self.nodal_displacements
        total_strain_energy = 0.5 * (u.T @ (K @ u))

        print(
            f"\n- Total strain energy in the system (from global computation): {total_strain_energy}"
        )

    # -----------------------------------------------------------------------------
    # TrussPlotter
    # -----------------------------------------------------------------------------
    def undeformed_mesh(self) -> None:
        """TrussPlotter: Plots the undeformed mesh structure."""
        # Only show the plot if Show_TrussPlot is not set to "0"
        if os.getenv("Show_TrussPlot", "1") != "0":
            self._plot_mesh(self.mesh.points, is_deformed=False)

    def deformed_mesh(self) -> None:
        """TrussPlotter: Plots the deformed mesh structure using the nodal displacements."""
        if self.nodal_displacements is None:
            raise ValueError("Displacement field (U) is not provided.")
        # Only show the plot if Show_TrussPlot is not set to "0"
        if os.getenv("Show_TrussPlot", "1") != "0":
            points = self._add_displacement(
                self.nodal_displacements, self.magnification
            )
            self._plot_mesh(points, is_deformed=True)

    def _add_displacement(self, U: np.ndarray, magnification: float) -> np.ndarray:
        """TrussPlotter: Applies the magnified displacement to each node in the mesh."""
        # Using broadcasting to reshape and apply magnification
        return self.mesh.points + U.reshape(-1, 2) * magnification

    def _plot_mesh(self, points: np.ndarray, is_deformed: bool) -> None:
        """TrussPlotter: Helper function to plot the mesh, either undeformed or deformed."""
        title = "deformed" if is_deformed else "undeformed"
        fig, axes = plt.subplots()
        axes.set_aspect("equal")
        fig.suptitle(title.capitalize() + " Mesh")

        if is_deformed:
            # Plot the undeformed mesh in light gray for context
            self._draw(
                self.mesh.points,
                self.mesh.element_connectivity,
                axes,
                color="lightgray",
            )
            # Plot the deformed mesh on top in red
            self._draw(points, self.mesh.element_connectivity, axes, color="red")
        else:
            # Plot only the undeformed mesh
            self._draw(points, self.mesh.element_connectivity, axes, color="black")

        self._add_node_label(points, axes)
        self._add_element_label(points, self.mesh.element_connectivity, axes)
        plt.tight_layout()
        plt.show()

    def _draw(
        self,
        points: np.ndarray,
        element_connectivity: list[list[int]] | np.ndarray,
        axes: matplotlib.axes.Axes,
        color: str,
        marker_color: str = "red",
    ) -> None:
        """TrussPlotter: Draws nodes and edges for the mesh."""
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        # Scatter plot for nodes and lines for elements
        axes.scatter(points[:, 0], points[:, 1], c=marker_color, alpha=0.3, marker="o")
        for node1, node2 in element_connectivity:
            x_coords, y_coords = points[[node1, node2], 0], points[[node1, node2], 1]
            axes.add_line(Line2D(x_coords, y_coords, linewidth=1.0, color=color))

    def _add_node_label(self, points: np.ndarray, axes: matplotlib.axes.Axes) -> None:
        """TrussPlotter: Adds labels to each node in the plot."""
        for idx, (x, y) in enumerate(points):
            axes.text(x, y, str(idx), color="b", size=10)

    def _add_element_label(
        self,
        points: np.ndarray,
        elements: list[list[int]] | np.ndarray,
        axes: matplotlib.axes.Axes,
    ) -> None:
        """TrussPlotter: Adds labels to each element in the plot."""
        for idx, (node1, node2) in enumerate(elements):
            x1, y1 = points[node1]
            x2, y2 = points[node2]
            # Position labels slightly off-center between nodes
            x_mid, y_mid = 0.6 * x1 + 0.4 * x2, 0.6 * y1 + 0.4 * y2
            axes.text(x_mid, y_mid, str(idx), color="g", size=10)
