#!/usr/bin/env python
"""
Module defining the PostProcessor class.

Created: 2025/10/18 18:03:29
Last modified: 2026/02/19 17:59:38
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
        number_elements: np.ndarray,
        alpha: float,
        alpha_counter: int,
        alpha_values: np.ndarray,
        E: float,
        A: float,
        P: float,
        magnification_factor: float = 0.0,
    ):
        self.mesh = mesh
        self.element_properties = element_properties
        self.global_stiffness_matrix = global_stiffness_matrix
        self.nodal_displacements = nodal_displacements
        self.number_elements = number_elements
        self.alpha = alpha
        self.alpha_counter = alpha_counter
        self.alpha_values = alpha_values
        self.E = E
        self.A = A
        self.P = P
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

    def compute_strain_energy_global(self, i: int, epsilon_h: np.ndarray) -> np.ndarray:
        """
        Computes the total strain energy using the global solution (U = 0.5 * u^T * K * u).

        Returns:
            epsilon_h.
        """

        number_elements = self.number_elements
        alpha_counter = self.alpha_counter
        alpha_values = self.alpha_values
        K = self.global_stiffness_matrix
        u = self.nodal_displacements

        epsilon_h[alpha_counter, i] = 0.5 * (u.T @ (K @ u))

        # if alpha_counter == (alpha_values.size - 1):
        #     if i == (number_elements.size - 1):
        #         print(
        #             f"\n- Total strain energy in the system (from FEM global computation): epsilon_h = {epsilon_h}"
        #         )

        return epsilon_h

    def compute_strain_energy_analytical(
        self,
        i: int,
        epsilon: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the analytical strain energy for the pull-out of a bar with E and A parameters in a medium with stiffness k.
        Formula: epsilon = (P / (2.0 * alpha * E * A)) * (-np.exp(-20.0 * alpha) + 1.0).

        Returns:
            epsilon.
        """

        alpha = self.alpha
        alpha_counter = self.alpha_counter
        alpha_values = self.alpha_values
        E = self.E
        A = self.A
        P = self.P
        number_elements = self.number_elements

        epsilon[alpha_counter, i] = (P / (2.0 * alpha * E * A)) * (
            -np.exp(-20.0 * alpha) + 1.0
        )

        # if alpha_counter == (alpha_values.size - 1):
        #     if i == (number_elements.size - 1):
        #         print(
        #             f"\n- Analytical solution of strain energy in the system: epsilon = {epsilon}"
        #         )

        return epsilon

    def compute_relative_error_in_energy(
        self, i: int, epsilon: np.ndarray, epsilon_h: np.ndarray, e: np.ndarray
    ) -> np.ndarray:
        """
        Computes the relative error in the energy norm using the formula: e = np.sqrt((epsilon - epsilon_h) / epsilon).
        epsilon = analitical solution for strain energy
        epsilon_h = strain energy from FEM global computation

        Returns:
            e.
        """

        number_elements = self.number_elements
        alpha_counter = self.alpha_counter
        alpha_values = self.alpha_values

        e[alpha_counter, i] = (
            (epsilon[alpha_counter, i] - epsilon_h[alpha_counter, i])
            / epsilon[alpha_counter, i]
        ) ** 0.5

        # if alpha_counter == (alpha_values.size - 1):
        #     if i == (number_elements.size - 1):
        #         print(f"\n- Relative error in the energy norm: e = {e}")

        return e

    def plot_u_uh(
        self,
        num_elements: int,
        L: float,
    ) -> None:
        """
        Print of the analitical solution and numerical one vs nodes position,
        related to displacements.

        Returns:
            None.
        """

        u_h = self.nodal_displacements.flatten()
        nodes_position = np.linspace(0, L, num_elements + 1)
        u = -(self.P / (self.E * self.A * self.alpha)) * np.exp(
            -self.alpha * nodes_position
        )

        plt.plot(
            nodes_position,
            u,
            marker="^",
            linestyle="-",
            color="r",
            label="Analytical Solution u",
        )
        plt.plot(
            nodes_position,
            u_h,
            marker="o",
            linestyle="-",
            label="FEM Solution u_h",
        )

        plt.xlabel("Position x")
        plt.ylabel("Displacement u(x) and u_h(x)")
        plt.title(
            f"Comparison: Analytical vs FEM (α = {self.alpha} and number of elements = {num_elements})"
        )
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.7)
        plt.show()

    def plot_solution(
        self,
        epsilon: np.ndarray,
        epsilon_h: np.ndarray,
    ) -> None:
        """
        Print of the analitical solution and numerical one vs number of elements,
        related to energy.

        Returns:
            None.
        """

        number_elements = self.number_elements
        alpha_values = self.alpha_values

        for i in range(alpha_values.size):
            plt.plot(
                number_elements,
                epsilon[i, :].flatten(),
                marker="^",
                linestyle="-",
                color="r",
                label=f"epsilon for alpha = {alpha_values[i]}",
            )
            plt.plot(
                number_elements,
                epsilon_h[i, :].flatten(),
                marker="o",
                linestyle="-",
                label=f"epsilon_h for alpha = {alpha_values[i]}",
            )

        plt.xlabel("Number of elements")
        plt.ylabel("Energy")
        plt.title(
            f"Solution of strain energy vs number of elements for α = {alpha_values}"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_e(
        self,
        e: np.ndarray,
        h: np.ndarray,
    ) -> None:
        """
        Print of the relative error in energy for different number of elements.

        Returns:
            None.
        """

        number_elements = self.number_elements
        alpha_values = self.alpha_values

        for i in range(alpha_values.size):
            plt.plot(
                number_elements,
                e[i, :].flatten(),
                marker="o",
                linestyle="-",
                label=f"alpha = {alpha_values[i]}",
            )
        plt.xlabel("Number of elements")
        plt.ylabel("Relative error in energy")
        plt.title(
            f"Relative error in energy in function of number of elements for α = {alpha_values}"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

        for i in range(alpha_values.size):
            plt.plot(
                h,
                e[i, :].flatten(),
                marker="x",
                linestyle="-",
                label=f"alpha = {alpha_values[i]}",
            )
        plt.xlabel("Element size h")
        plt.ylabel("Relative error in energy")
        plt.title(
            f"Relative error in energy in function of element size h for α = {alpha_values}"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_e_grading(
        self,
        e: np.ndarray,
        dofs: np.ndarray,
    ) -> None:
        """
        Print of the relative error in energy for different number of elements and DOFs.

        Returns:
            None.
        """

        number_elements = self.number_elements
        alpha_values = self.alpha_values

        for i in range(alpha_values.size):
            plt.plot(
                number_elements,
                e[i, :].flatten(),
                marker="o",
                linestyle="-",
                label=f"alpha = {alpha_values[i]}",
            )
        plt.xlabel("Number of elements")
        plt.ylabel("Relative error in energy")
        plt.title(
            f"Relative error in energy norm in function of number of elements for α = {alpha_values} - WITH MESH GRADING"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plotting against DOFs instead of h
        for i in range(alpha_values.size):
            plt.plot(
                dofs,
                e[i, :].flatten(),
                marker="x",
                linestyle="-",
                label=f"alpha = {alpha_values[i]}",
            )
        plt.xlabel("Degrees of Freedom (DOFs)")
        plt.ylabel("Relative error in energy")
        plt.title(
            f"Relative error in energy in function of DOFs for α = {alpha_values} - WITH MESH GRADING"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_convergence_rate(
        self,
        e: np.ndarray,
        h: np.ndarray,
        cond_numbers: np.ndarray,
    ) -> None:
        """
        Print of the covergence rate in energy.

        Returns:
            None.
        """

        number_elements = self.number_elements
        alpha_values = self.alpha_values

        for i in range(alpha_values.size):
            plt.loglog(
                number_elements,
                e[i, :].flatten(),
                marker="o",
                linestyle="-",
                label=f"Error for alpha = {alpha_values[i]}",
            )

            plt.loglog(
                number_elements,
                cond_numbers[i, :].flatten(),
                marker="*",
                linestyle="-",
                label=f"Cond. number for alpha = {alpha_values[i]}",
            )
        plt.xlabel("Number of elements (Log scale)")
        plt.ylabel("Error in energy and conditioning numbers (Log scale)")
        plt.title(
            f"Error and Conditioning number for α = {alpha_values} in log-log scale"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

        for i in range(alpha_values.size):
            plt.loglog(
                h,
                e[i, :].flatten(),
                marker="x",
                linestyle="-",
                label=f"Error for alpha = {alpha_values[i]}",
            )
            plt.plot(
                h,
                cond_numbers[i, :].flatten(),
                marker="*",
                linestyle="-",
                label=f"Cond. number for alpha = {alpha_values[i]}",
            )
        plt.xlabel("Element size h (Log scale)")
        plt.ylabel("Error in energy and conditioning numbers (Log scale)")
        plt.title(
            f"Error and Conditioning number for α = {alpha_values} in log-log scale"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_convergence_rates(
        self, e: np.ndarray, x_array: np.ndarray, is_dofs: bool = False
    ) -> None:
        """
        Calculates the asymptotic convergence rates for multiple alpha values.
        It iterates over the rows of the error matrix.

        Parameters:
        e       (np.ndarray): 2D array of relative errors (rows: alpha values, columns: mesh refinements).
        x_array (np.ndarray): 1D array of element sizes (h) or Degrees of Freedom (DOFs).
        is_dofs (bool): True if x_array contains DOFs, False if it contains h.

        Returns:
        list[float]: A list containing the calculated asymptotic convergence rate for each alpha.
        """
        convergence_rates = []
        num_alphas = e.shape[0]
        alpha_values = self.alpha_values

        for i in range(num_alphas):
            # Extract the 1D error array for the current alpha (i-th row)
            e_row = e[i, :]

            # Filter out uncomputed values (zeros) or NaNs to prevent math errors
            valid_mask = (e_row > 0) & (~np.isnan(e_row))
            valid_e = e_row[valid_mask]
            valid_x = x_array[valid_mask]

            if len(valid_e) < 2:
                # Append a NaN value and print the failure explicitly
                convergence_rates.append(float("nan"))
                print(
                    f"- Convergence rate for alpha = {alpha_values[i]}: NaN (insufficient data)"
                )
                continue

            # Find the index of the minimum error before round-off errors dominate
            min_idx = int(np.argmin(valid_e))

            if min_idx >= 2:
                # Rigorous asymptotic calculation BEFORE the loss of precision
                idx2 = min_idx - 1
                idx1 = min_idx - 2
            else:
                # Fallback to the last two available valid points
                idx2 = -1
                idx1 = -2

            e1, e2 = valid_e[idx1], valid_e[idx2]
            x1, x2 = valid_x[idx1], valid_x[idx2]

            # Calculate log-log slope
            slope = (np.log(e2) - np.log(e1)) / (np.log(x2) - np.log(x1))

            # The slope is inverted if calculated with respect to DOFs
            rate = -slope if is_dofs else slope
            convergence_rates.append(rate)

            # Print the successfully calculated rate
            print(f"- Convergence rate for alpha = {alpha_values[i]}: {rate:.4f}")

        return None

    def plot_convergence_rate_grading(
        self,
        e: np.ndarray,
        dofs: np.ndarray,
        cond_numbers: np.ndarray,
    ) -> None:
        """
        Print of the convergence rate in energy vs DOFs (Log-Log scale).

        Returns:
            None.
        """

        number_elements = self.number_elements
        alpha_values = self.alpha_values

        for i in range(alpha_values.size):
            plt.loglog(
                number_elements,
                e[i, :].flatten(),
                marker="o",
                linestyle="-",
                label=f"Error for alpha = {alpha_values[i]}",
            )

            plt.loglog(
                number_elements,
                cond_numbers[i, :].flatten(),
                marker="*",
                linestyle="-",
                label=f"Cond. number for alpha = {alpha_values[i]}",
            )
        plt.xlabel("Number of elements (Log scale)")
        plt.ylabel("Error in energy and conditioning numbers (Log scale)")
        plt.title(
            f"Error and Conditioning number for α = {alpha_values} in log-log scale - WITH MESH GRADING"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plotting against DOFs instead of h
        for i in range(alpha_values.size):
            plt.loglog(
                dofs,
                e[i, :].flatten(),
                marker="x",
                linestyle="-",
                label=f"Error for alpha = {alpha_values[i]}",
            )
            plt.loglog(
                dofs,
                cond_numbers[i, :].flatten(),
                marker="*",
                linestyle="-",
                label=f"Cond. number for alpha = {alpha_values[i]}",
            )
        plt.xlabel("Degrees of Freedom (DOFs) (Log scale)")
        plt.ylabel("Error in energy and conditioning numbers (Log scale)")
        plt.title(
            f"Error and Conditioning number for α = {alpha_values} in log-log scale - WITH MESH GRADING"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

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
