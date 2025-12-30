#!/usr/bin/env python
"""
Module defining the Model class for finite element analysis.

Created: 2025/10/28 01:25:31
Last modified: 2025/11/24 22:35:25
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import os

from .boundary_conditions import BoundaryConditions
from .element_compatibility import validate_element_problem_compatibility
from .element_properties import ElementProperties, validate_mesh_and_element_properties
from .elements.element_registry import ELEMENTS_THAT_REQUIRE_MATERIAL
from .materials import MaterialProperties
from .mesh import Mesh
from .problem import Problem
from .setup_helpers import setup_dof_space_for_problem
from .solution import Solution


class ModelValidationError(Exception):
    """Raised when model validation fails."""

    pass


class Model:
    """Unified container for a complete finite element model.

    The Model class combines all components needed for FEA:
    - Mesh: geometry and topology
    - Problem: physics type and spatial dimension
    - DOF space: degree of freedom management
    - Element properties: material and element definitions
    - Boundary conditions: Dirichlet and Neumann conditions

    The DOF space is automatically configured based on the problem type.

    Attributes:
        mesh: Mesh object containing nodes, elements, and node sets
        problem: Problem instance (physics + dimension)
        dof_space: DOF space (automatically created from problem)
        element_properties: Registry of element property definitions
        boundary_conditions: Boundary conditions manager
    """

    def __init__(self, mesh: Mesh, problem: Problem):
        """Initialize a model with mesh and problem.

        Args:
            mesh: Mesh object
            problem: Problem instance (e.g., Problem(Physics.MECHANICS, Dimension.D1))

        Example:
            problem = Problem(Physics.MECHANICS, Dimension.D1)
            model = Model(mesh, problem)
        """
        self.mesh = mesh
        self.problem = problem
        self.materials: MaterialProperties = {}

        # Automatically create and configure DOF space based on problem
        self.dof_space = setup_dof_space_for_problem(problem, mesh.num_nodes)

        # Initialize element properties (to be set by user)
        self.element_properties: ElementProperties = {}

        # Initialize boundary conditions
        self.bc = BoundaryConditions(self.dof_space, self.mesh)

    def set_element_properties(self, element_properties: ElementProperties) -> None:
        """Set element properties for the model.

        This method performs comprehensive validation:
        1. Validates mesh and element properties compatibility
        2. Validates element-problem compatibility

        Args:
            element_properties: Dictionary of element property definitions

        Raises:
            ModelValidationError: If validation fails

        Example:
            props = make_element_properties([
                ("steel_bar", ElementProperty(
                    kind="bar_1D",
                    params={"A": 0.01},
                    material="steel",
                )),
            ])
            model.set_element_properties(props)
        """
        self.element_properties = element_properties

        # Validation 1: Mesh and element properties
        try:
            validate_mesh_and_element_properties(self.mesh, self.element_properties)
        except Exception as e:
            raise ModelValidationError(
                f"Mesh and element properties validation failed: {e}"
            ) from e

        # Validation 2: Element-problem compatibility
        # Extract unique element kinds from the properties
        element_kinds = set()
        for label in self.mesh.element_property_labels:
            if label in self.element_properties:
                elem_prop = self.element_properties[label]
                element_kinds.add(elem_prop.kind)

        # Validate compatibility
        incompatible = validate_element_problem_compatibility(
            list(element_kinds), self.problem
        )

        if incompatible:
            raise ModelValidationError(
                f"Element(s) {incompatible} are not compatible with problem "
                f"{self.problem.physics.name} ({self.problem.dimension.value}D). "
                f"Cannot use these elements in this problem configuration."
            )

        # Validation 3: Material presence for required elements
        unique_labels = set(self.mesh.element_property_labels)

        for label in unique_labels:
            elem_prop = self.element_properties[label]

            if elem_prop.kind in ELEMENTS_THAT_REQUIRE_MATERIAL:
                if elem_prop.material is None:
                    raise ModelValidationError(
                        f"Element property '{label}' of type '{elem_prop.kind}' "
                        f"requires a material but none was provided.\n"
                        f"Use: ElementProperty(material='name')."
                    )

            if (
                elem_prop.kind not in ELEMENTS_THAT_REQUIRE_MATERIAL
                and elem_prop.material is not None
            ):
                print(
                    f"Warning: Element property '{label}' of type '{elem_prop.kind}' "
                    f"has a material assigned, but this element type does not use materials."
                )

    def set_materials(self, materials: MaterialProperties):
        self.materials = materials

    def __str__(self) -> str:
        """Return a concise, human-readable summary of the model."""
        return (
            f"\n"
            f"Model Summary:\n"
            f"  Problem Type: {self.problem}\n"
            f"  Mesh: {self.mesh.num_nodes} nodes, {self.mesh.num_elements} elements\n"
            f"  Degrees of Freedom: {self.dof_space.total_dofs}\n"
            f"  Node Sets: {len(self.mesh.node_sets)}"
        )

    def set_solution(self, solution: Solution) -> None:
        """Store the solution (used later for export and post-processing)."""
        self._solution = solution

    @property
    def solution(self) -> Solution:
        """Access stored solution."""
        return self._solution

    # ----------------------------------------------------------------------
    # VTK SAVE UTILITIES
    # ----------------------------------------------------------------------
    def _resolve_save_path(self, filename: str, save_to: str | None) -> str:
        """
        Resolve final export path:
        - if filename is absolute -> return as is
        - else -> save relative to the user script that executed pyFEM
        - if save_to is provided, create and append folder automatically
        """
        # Case 1: absolute path -> use as given
        if os.path.isabs(filename):
            return filename

        # Case 2: relative -> detect the script that imported pyFEM
        import __main__

        try:
            base = os.path.dirname(os.path.abspath(__main__.__file__))
        except AttributeError:
            # interactive session (IPython/Jupyter) fallback
            base = os.getcwd()

        # Optionally create user-specified subfolder
        if save_to is not None:
            base = os.path.join(base, save_to)
            os.makedirs(base, exist_ok=True)

        return os.path.join(base, filename)

    def save_vtk(
        self,
        filename: str,
        *,
        displacements: bool = False,
        stresses: bool = False,
        save_to: str | None = None,
    ) -> None:
        """Export VTK/VTU file to the correct location."""
        if not hasattr(self, "_solution") or self._solution is None:
            raise RuntimeError("No solution found. Run a Step before exporting.")

        fullpath = self._resolve_save_path(filename, save_to)
        from .vtk_export import VTKWriter  # lazy import

        writer = VTKWriter(self, self._solution)
        writer.write(fullpath, displacements=displacements, stresses=stresses)
