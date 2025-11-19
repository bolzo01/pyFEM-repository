#!/usr/bin/env python
"""
Module defining the PostProcessor class.

Created: 2025/10/18 18:03:29
Last modified: 2025/11/17 23:57:05
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import os

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import sparse

from .element_properties import param
from .elements.element_registry import ELEMENTS_THAT_REQUIRE_MATERIAL
from .materials import LinearElastic1D
from .model import Model
from .solution import Solution


class PostProcessor:
    """Handles post-processing computations for finite element analysis.

    Computes strain energy and other derived quantities from FEA solutions.
    """

    def __init__(
        self,
        model: Model,
        solution: Solution,
        global_stiffness_matrix: np.ndarray | sparse.spmatrix | None,
        magnification_factor: float = 0.0,
    ):
        self.model = model
        self.mesh = model.mesh
        self.element_properties = model.element_properties
        self.solution = solution
        self.nodal_displacements = solution.nodal_displacements
        self.global_stiffness_matrix = global_stiffness_matrix
        self.magnification = magnification_factor

    def compute_strain_energy_local(self) -> None:
        """
        Computes the total strain energy by summing the strain energy of each element.

        Returns:
            None.
        """

        num_elements = self.mesh.num_elements

        total_strain_energy = 0.0
        for element_index in range(num_elements):
            label = self.mesh.element_property_labels[element_index]
            elem_prop = self.element_properties[label]
            kind = elem_prop.kind

            if kind == "spring_1D":
                k_spring = param(elem_prop, "k", float)
                n1, n2 = self.mesh.element_connectivity[element_index]
                u1 = self.nodal_displacements[n1]
                u2 = self.nodal_displacements[n2]
                energy = 0.5 * k_spring * (u2 - u1) ** 2

            elif kind == "bar_1D":
                # Retrieve the material
                material = self._resolve_material(elem_prop, label)

                if not isinstance(material, LinearElastic1D):
                    raise TypeError(
                        f"{kind} element requires LinearElastic1D material, "
                        f"but got {type(material).__name__}"
                    )

                E = float(material.E)
                A = param(elem_prop, "A", float)

                n1, n2 = self.mesh.element_connectivity[element_index]
                L = abs(self.mesh.points[n2] - self.mesh.points[n1])
                u1 = self.nodal_displacements[n1]
                u2 = self.nodal_displacements[n2]
                strain = (u2 - u1) / L
                stress = E * strain
                energy = 0.5 * stress * strain * A * L

            else:
                raise NotImplementedError(
                    f"Strain energy not implemented for element kind '{kind}'"
                )

            print(f"- Strain energy in element {element_index}: {energy}")

            total_strain_energy += energy

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

    def _resolve_material(self, elem_prop, label: str):
        """
        Return the Material instance if the element requires a material.
        Return None for material-free elements (e.g. spring_1D).
        """
        kind = elem_prop.kind

        # ------------------------------------------------------------
        # 1. Elements that DO NOT use material properties
        # ------------------------------------------------------------
        if kind not in ELEMENTS_THAT_REQUIRE_MATERIAL:
            return None  # <-- the fix: springs skip material resolution entirely

        # ------------------------------------------------------------
        # 2. Elements that DO require a material
        # ------------------------------------------------------------
        material_name = elem_prop.material

        # Fallback on meta["material"]
        if material_name is None:
            raw = elem_prop.meta.get("material")
            if isinstance(raw, str):
                material_name = raw
            else:
                raise ValueError(
                    f"Element property '{label}' of type '{kind}' "
                    f"requires a material but none was provided."
                )

        # Verify existence
        if material_name not in self.model.materials:
            raise ValueError(
                f"Material '{material_name}' not found in model.materials "
                f"for element '{label}'."
            )

        return self.model.materials[material_name]

    def compute_element_stresses(self) -> list:
        """
        Compute stresses for each element at their Gauss points.

        Returns
        -------
        list
            A list where stresses[e] = array of stresses for element e.
        """
        from .elements.element_registry import create_element

        stresses = []
        mesh = self.model.mesh
        U = self.solution.nodal_displacements

        for e in range(mesh.num_elements):
            # Element property
            label = mesh.element_property_labels[e]
            elem_prop = self.model.element_properties[label]

            # Element instance
            element = create_element(elem_prop)

            # Node coordinates
            elem_nodes = mesh.element_connectivity[e]
            x_nodes = mesh.points[elem_nodes]

            # Element displacement vector
            dof_map = self.model.dof_space.get_dof_mapping(elem_nodes)
            u_nodes = U[dof_map]

            # Material
            material = self._resolve_material(elem_prop, label)

            # Stress at Gauss points (or analytical)
            sigma = element.compute_stress(material, x_nodes, u_nodes)

            stresses.append(sigma)

        # Store in the Solution object
        self.solution.element_stresses = stresses

        print("Computed stresses for each element.")
        return stresses

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
