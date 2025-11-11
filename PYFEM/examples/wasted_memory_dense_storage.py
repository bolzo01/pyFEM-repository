#!/usr/bin/env python
"""
The inefficiency of dense storage for stiffness matrices.

This script demonstrates how rapidly a dense matrix become
inefficient when assembling and solving structural stiffness matrices for
increasingly large truss systems. It uses the `brown_truss` module to
generate truss problems with varying numbers of bays, solves each system,
and records key performance metrics including:

- The percentage of zero entries in the global stiffness matrix (K),
- The computational time required to solve the system, and
- The memory footprint of the dense matrix representation.

Created: 2025/10/29 18:26:04
Last modified: 2025/11/08 10:44:04
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import os
import time
from typing import Generator

import brown_truss
import matplotlib.pyplot as plt
from matplotlib import animation


def data_gen() -> Generator[tuple[int, float, float, float], None, None]:
    """Generates data points for each step."""
    for bays in range(5, 1000, 25):
        print(f"Processing bays = {bays}")
        start_time = time.time()
        _, dofs, sparsity_percentage, matrix_size_bytes, _ = brown_truss.main(
            bays,
            use_sparse=False,
        )
        total_time = time.time() - start_time
        matrix_size_mib = matrix_size_bytes / (1024 * 1024)  # Convert bytes to MiB
        yield dofs, sparsity_percentage, total_time, matrix_size_mib


def main() -> None:
    # Set environment variable to control plot display behavior
    os.environ["Show_TrussPlot"] = "0"

    # Create plot and initialize data
    figure, axes_zero_entries, axes_total_time, axes_matrix_size = setup_figure()

    # Initialize data storage
    x_data, y_data_zero_entries, y_data_total_time, y_data_matrix_size = (
        [],
        [],
        [],
        [],
    )

    def init() -> tuple[plt.Line2D, plt.Line2D, plt.Line2D]:
        """Initialize the three plots and clear data."""
        for ax, line, y_data in zip(
            (axes_zero_entries, axes_total_time, axes_matrix_size),
            (line_zero_entries, line_total_time, line_matrix_size),
            (y_data_zero_entries, y_data_total_time, y_data_matrix_size),
            strict=False,
        ):
            ax.set_xlim(0, 200)
            y_data.clear()
            line.set_data(x_data, y_data)

        return line_zero_entries, line_total_time, line_matrix_size

    def update(
        data: tuple[int, float, float, float],
    ) -> tuple[plt.Line2D, plt.Line2D, plt.Line2D]:
        """Updates all plots with new data."""
        dofs, zero_entries, total_time, matrix_size_mib = data

        x_data.append(dofs)
        y_data_zero_entries.append(zero_entries)
        y_data_total_time.append(total_time)
        y_data_matrix_size.append(matrix_size_mib)

        for ax, line, y_data in zip(
            (axes_zero_entries, axes_total_time, axes_matrix_size),
            (line_zero_entries, line_total_time, line_matrix_size),
            (y_data_zero_entries, y_data_total_time, y_data_matrix_size),
            strict=False,
        ):
            if dofs >= ax.get_xlim()[1]:
                ax.set_xlim(0, dofs * 1.5)
            ax.set_ylim(min(y_data) * 0.9, max(y_data) * 1.1)
            line.set_data(x_data, y_data)

        return line_zero_entries, line_total_time, line_matrix_size

    # Initialize lines
    line_zero_entries = axes_zero_entries.plot(
        [], [], "-bx", label="Zero entries in K (%)"
    )[0]
    line_total_time = axes_total_time.plot([], [], "-g", label="Total time (s)")[0]
    line_matrix_size = axes_matrix_size.plot([], [], "-r", label="Matrix Size (MiB)")[0]

    # Animate the data
    _ = animation.FuncAnimation(
        figure, update, frames=data_gen, init_func=init, repeat=False, save_count=200
    )
    plt.show()
    plt.close(figure)
    plt.close("all")

    # Clean up the environment variable
    del os.environ["Show_TrussPlot"]


def setup_figure() -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    """Sets up the figure and axes."""
    figure = plt.figure(figsize=(14, 8))
    axes_zero_entries = figure.add_subplot(1, 2, 1)
    axes_total_time = figure.add_subplot(2, 2, 2)
    axes_matrix_size = figure.add_subplot(2, 2, 4)

    for ax, title, ylabel in zip(
        (axes_zero_entries, axes_total_time, axes_matrix_size),
        [
            " ",
            " ",
            " ",
        ],
        [
            "Zero entries in K (%)",
            "Total time (s)",
            "Matrix size (MiB)",
        ],
        strict=False,
    ):
        ax.set_title(title)
        ax.set_xlabel("System size (DOFs)")
        ax.set_ylabel(ylabel)
        ax.grid()

    return figure, axes_zero_entries, axes_total_time, axes_matrix_size


if __name__ == "__main__":
    main()
