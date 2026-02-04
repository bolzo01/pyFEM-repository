#!/usr/bin/env python
"""
VTK Writer for finite element results.

Created: 2025/11/23 22:15:26
Last modified: 2026/02/04 17:34:46
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
        temperatures: bool = False,
        save_to: str | None = None,
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

        # Original coordinates, normalized to (npoints, dim)
        orig_points = np.asarray(self.mesh.points, dtype=float)

        if orig_points.ndim == 1:
            orig_points = orig_points.reshape(-1, 1)

        # Default behavior: extend to 3D for all problems
        points = orig_points
        if points.shape[1] == 1:
            points = np.column_stack([points, np.zeros((len(points), 2))])
        elif points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(len(points))])

        # Build connectivity grouped by *VTK/meshio* cell type
        cells: list[tuple[str, np.ndarray]] = []

        # Group connectivity by meshio cell type
        celltype_groups: dict[str, list[list[int]]] = defaultdict(list)

        # Map pyFEM element kinds -> meshio cell types
        element_kind_to_meshio = {
            "bar_1D": "line",
            "bar3_1D": "line3",
            "bar_1D_heat": "line",
            "triangle": "triangle",
            "triangle_heat": "triangle",
            "tetra": "tetra",
        }

        for label, conn in zip(
            self.mesh.element_property_labels, self.mesh.element_connectivity
        ):
            elem_kind = self.model.element_properties[label].kind

            try:
                cell_type = element_kind_to_meshio[elem_kind]
            except KeyError:
                raise ValueError(
                    f"VTK export: no meshio cell type mapping defined for "
                    f"element kind '{elem_kind}'. Please extend "
                    f"'element_kind_to_meshio' in VTKWriter."
                )

            celltype_groups[cell_type].append(conn)

        for cell_type, conn_list in celltype_groups.items():
            cells.append((cell_type, np.array(conn_list, dtype=int)))

        # Point data
        point_data: dict[str, np.ndarray] = {}

        if displacements and self.solution.nodal_displacements is not None:
            U = self.solution.nodal_displacements

            # Normalize original mesh points to get dimension safely
            pts = np.asarray(self.model.mesh.points, dtype=float)
            if pts.ndim == 1:
                pts = pts.reshape(-1, 1)
            dim = pts.shape[1]

            U = U.reshape(-1, dim)

            if dim == 1:
                U = np.column_stack([U, np.zeros((len(U), 2))])
            elif dim == 2:
                U = np.column_stack([U, np.zeros(len(U))])

            point_data["Displacement"] = U.astype(float)

        # Temperatures as scalar point data
        if (
            temperatures
            and self.solution is not None
            and self.solution.nodal_displacements is not None
        ):
            # For HEAT_TRANSFER, each node has one temperature dof,
            # so nodal_displacements is the temperature vector.
            if (
                temperatures
                and self.solution is not None
                and self.solution.nodal_displacements is not None
            ):
                T = self.solution.nodal_displacements.reshape(-1)
                point_data["Temperature"] = T.astype(float)

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
            binary=True,
        )

        print(f"VTK file written: {filename}")
