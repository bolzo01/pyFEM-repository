#!/usr/bin/env python
"""
Module defining the Model class for finite element analysis.

Created: 2025/10/28 01:25:31
Last modified: 2025/10/30 00:35:03
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from .boundary_conditions import BoundaryConditions
from .element_properties import ElementProperties
from .mesh import Mesh
from .problem import Problem
from .setup_helpers import setup_dof_space_for_problem


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

        # Automatically create and configure DOF space based on problem
        self.dof_space = setup_dof_space_for_problem(problem, mesh.num_nodes)

        # Initialize element properties (to be set by user)
        self.element_properties: ElementProperties = {}

        # Initialize boundary conditions
        self.bc = BoundaryConditions(self.dof_space, self.mesh)

    def set_element_properties(self, element_properties: ElementProperties) -> None:
        """Set element properties for the model.

        Args:
            element_properties: Dictionary of element property definitions

        Example:
            props = make_element_properties([
                ("steel", ("bar_1D", {"E": 200e9, "A": 0.01})),
            ])
            model.set_element_properties(props)
        """
        self.element_properties = element_properties

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
