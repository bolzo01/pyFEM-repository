#!/usr/bin/env python
"""
VTK Writer for finite element results.

Created: 2025/11/23 22:15:26
Last modified: 2025/11/24 01:39:00
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from collections import defaultdict

import meshio  # type: ignore
import numpy as np

from .model import Model
from .solution import Solution


class VTKWriter:
    """
    Export utility to VTK/VTU files.

    The actual format is inferred from the filename extension:
      - *.vtk -> legacy VTK
      - *.vtu -> XML unstructured grid
    """

    def __init__(self, model: Model, solution: Solution):
        self.model = model
        self.mesh = model.mesh
        self.solution = solution

    def write(
        self,
        filename: str,
        displacements: bool = False,
        stresses: bool = False,
    ) -> None:
        """
        Write mesh and (optionally) result fields to a VTK/VTU file.

        Args:
            model: pyFEM Model (provides mesh, dof_space, materials, etc.)
            solution: Solution object with displacements and, optionally, stresses.
            settings: VTKOutputSettings describing what to export.

        Notes:
            - File format is chosen by the extension of `settings.filename`:
                *.vtk -> legacy VTK
                *.vtu -> XML VTU
            - For displacements we currently support mechanical DOFs: U_X, U_Y, U_Z.
            - For stresses, `solution.element_stresses` should contain stress vectors
            in Voigt notation per element:
                * 1D: [sigma_xx]
                * 2D (plane stress/strain): [sigma_xx, sigma_yy, tau_xy]
                * 2D (with out-of-plane): [sigma_xx, sigma_yy, sigma_zz, tau_xy]
                * 3D: [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_zx]
            - In ParaView, you'll see separate scalar fields: sigma_xx, sigma_yy, tau_xy, etc.
        """
        points = self.mesh.points.astype(float)

        # VTK expects 3D coordinates. Extend 1D/2D to 3D.
        if points.shape[1] == 1:
            points = np.column_stack([points, np.zeros((len(points), 2))])
        elif points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(len(points))])

        # Build connectivity grouped by element type
        cells: list[tuple[str, np.ndarray]] = []

        # Group connectivity by element type
        etype_groups: dict[str, list[list[int]]] = defaultdict(list)

        for label, conn in zip(
            self.mesh.element_property_labels, self.mesh.element_connectivity
        ):
            element_type = self.model.element_properties[label].kind
            etype_groups[element_type].append(conn)

        for etype, conn_list in etype_groups.items():
            cells.append((etype, np.array(conn_list, dtype=int)))

        # Point data
        point_data: dict[str, np.ndarray] = {}

        if displacements and self.solution.nodal_displacements is not None:
            U = self.solution.nodal_displacements
            dim = self.model.mesh.points.shape[1]
            U = U.reshape(-1, dim)

            if dim == 1:
                U = np.column_stack([U, np.zeros((len(U), 2))])
            elif dim == 2:
                U = np.column_stack([U, np.zeros(len(U))])

            point_data["Displacement"] = U.astype(float)

        # Cell data (stresses)
        cell_data: dict[str, list[np.ndarray]] = {}

        if stresses and self.solution.element_stresses is not None:
            # Convert to array per group
            all_stresses = [np.asarray(s) for s in self.solution.element_stresses]
            stress_array = np.vstack(all_stresses)

            # Determine number of stress components
            n_components = stress_array.shape[1] if stress_array.ndim > 1 else 1

            # Create named stress fields based on dimensionality
            if n_components == 1:
                # 1D: axial stress
                cell_data["sigma_xx"] = [stress_array[:, 0]]
            elif n_components == 3:
                # 2D: sigma_xx, sigma_yy, tau_xy
                cell_data["sigma_xx"] = [stress_array[:, 0]]
                cell_data["sigma_yy"] = [stress_array[:, 1]]
                cell_data["tau_xy"] = [stress_array[:, 2]]
            elif n_components == 4:
                # 2D with out-of-plane: sigma_xx, sigma_yy, sigma_zz, tau_xy
                cell_data["sigma_xx"] = [stress_array[:, 0]]
                cell_data["sigma_yy"] = [stress_array[:, 1]]
                cell_data["sigma_zz"] = [stress_array[:, 2]]
                cell_data["tau_xy"] = [stress_array[:, 3]]
            elif n_components == 6:
                # 3D: full stress tensor
                cell_data["sigma_xx"] = [stress_array[:, 0]]
                cell_data["sigma_yy"] = [stress_array[:, 1]]
                cell_data["sigma_zz"] = [stress_array[:, 2]]
                cell_data["tau_xy"] = [stress_array[:, 3]]
                cell_data["tau_yz"] = [stress_array[:, 4]]
                cell_data["tau_zx"] = [stress_array[:, 5]]
            else:
                raise ValueError(
                    f"Cannot define stress vector: got {n_components} components."
                )

        # Write via meshio
        meshio.write_points_cells(
            filename=filename,
            points=points,
            cells=cells,
            point_data=point_data if point_data else None,
            cell_data=cell_data if cell_data else None,
        )

        print(f"VTK file written: {filename}")
