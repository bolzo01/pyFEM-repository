#!/usr/bin/env python
"""
Module defining the Step class for analysis execution (similar to Abaqus step concept)

Created: 2025/11/16 10:24:33
Last modified: 2025/11/28 01:16:05
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import os
from dataclasses import dataclass, field
from enum import Enum, auto

from .model import Model
from .post_processor import PostProcessor
from .solution import Solution
from .solvers import LinearStaticSolver


class ProcedureType(Enum):
    """Analysis procedure types."""

    STATIC_LINEAR = auto()
    STATIC_LINEAR_INCREMENTAL = auto()
    # Future: STATIC_NONLINEAR, DYNAMIC_IMPLICIT, HEAT_TRANSFER, etc.


@dataclass
class ModelState:
    """
    Tracks solutions obtained across analysis steps.

    In a multi-step analysis (Abaqus-style), different steps may represent
    different load cases or analysis configurations. This class does not
    update the mechanical state of the model; rather, it stores the
    solutions produced by each step so they can be retrieved, compared,
    and post-processed.

    Typical usage:
        - Step 1 solves with one set of loads/BCs.
        - The user modifies the model.
        - Step 2 solves with the new conditions.
        - This class keeps both solutions available.

    Attributes:
        current_solution: The solution from the most recently executed step.
        step_solutions: Mapping from step names to their corresponding solutions.
        step_count: Total number of executed steps.

    Example:
        state = ModelState()
        state = step1.execute(model, state)
        state = step2.execute(model, state)

        # Compare load cases
        u1 = state.get_step_solution("LoadCase1").nodal_displacements
        u2 = state.get_step_solution("LoadCase2").nodal_displacements

    Note:
        Steps do not automatically modify or reset boundary conditions.
        The Model retains whatever BCs or loads were last assigned.
        Therefore, unless the user explicitly changes `model.bc` between steps,
        each step will solve with the same loading and constraints.
    """

    _current_solution: Solution | None = None
    step_solutions: dict[str, Solution] = field(default_factory=dict)
    step_count: int = 0

    @property
    def current_solution(self) -> Solution:
        """Return the most recent solution (never None once a step has run)."""
        if self._current_solution is None:
            raise RuntimeError(
                "No current solution is available. Execute a step first."
            )
        return self._current_solution

    def update_solution(self, step_name: str, solution: Solution) -> None:
        """
        Store the solution produced by a step and update the state bookkeeping.

        This method records the solution under the given step name, marks it as the
        current solution, and increments the internal step counter.

        Args:
            step_name: The name of the step that produced the solution.
            solution: The Solution object returned by that step.
        """
        self._current_solution = solution
        self.step_solutions[step_name] = solution
        self.step_count += 1

    def get_step_solution(self, step_name: str) -> Solution:
        """
        Return the solution associated with a specific step.

        Args:
            step_name: The name of the step whose solution is requested.

        Returns:
            The Solution object stored for that step.

        Raises:
            KeyError: If no solution has been recorded under the given step name.
        """

        if step_name not in self.step_solutions:
            available = ", ".join(self.step_solutions.keys())
            raise KeyError(
                f"No solution found for step '{step_name}'. "
                f"Available steps: {available if available else 'None'}"
            )
        return self.step_solutions[step_name]

    def has_step(self, step_name: str) -> bool:
        """
        Check whether a solution for the given step has been recorded.

        Args:
            step_name: The name of the step to check.

        Returns:
            True if a solution exists for that step; otherwise False.
        """
        return step_name in self.step_solutions

    def __str__(self) -> str:
        """Summary of model state."""
        if self.step_count == 0:
            status = "No steps executed"
            steps = "None"
        else:
            status = f"{self.step_count} step(s) executed"
            steps = ", ".join(self.step_solutions.keys())

        return f"Model State:\n  Status: {status}\n  Steps: {steps}"


@dataclass
class Step:
    """
    Represents an analysis step (Abaqus-style).

    A Step defines:
        - The analysis procedure to run (e.g., static linear).
        - The logic required to execute that procedure.
        - Optional post-processing operations.

    Steps follow the Abaqus organization model: an analysis can consist of
    multiple named steps, each representing a distinct load case or analysis
    configuration. However, this implementation performs independent linear
    static solves. No mechanical state (e.g., stresses, strains, plastic
    variables) is carried from one step to the next.

    Each Step operates on a Model, which contains mesh data, material
    properties, and boundary conditions. The user is responsible for
    modifying the Model between steps when different loading or constraint
    conditions are desired.

    Importantly:
        Boundary conditions and loads in the Model are not automatically
        reset between steps. If the user does not change them, subsequent
        steps will solve with the same conditions as the previous step.

    Attributes:
        name: Descriptive name for this step.
        procedure: Type of analysis procedure to execute.
        verbose: Whether to print progress information.
        _last_solver: Internal reference to the solver used for this step.

    Example:
        # Single-step analysis
        step = Step(
            name="ApplyLoad",
            procedure=ProcedureType.STATIC_LINEAR,
            verbose=True
        )
        model_state = ModelState()
        model_state = step.execute(model, model_state, use_sparse=False)

        # Multi-step analysis (two different load cases)
        step1 = Step(name="LoadCase1", procedure=ProcedureType.STATIC_LINEAR)
        step2 = Step(name="LoadCase2", procedure=ProcedureType.STATIC_LINEAR)

        state = ModelState()
        state = step1.execute(model, state)

        # Modify BCs for the second step
        # (BCs persist unless explicitly changed by the user)
        model.bc.registry._forces.clear()
        model.bc.apply_force(node, DOFType.U_X, new_force)

        state = step2.execute(model, state)


    ramp: optional user function ramp(i, n) -> float
        i = current increment number (1-based)
        n = total number of increments
        The function must return a load factor between 0 and 1.

        Examples:
            ramp=lambda i, n: i/n          # linear
            ramp=lambda i, n: (i/n)**2     # quadratic
            ramp=lambda i, n: np.sin(0.5*np.pi*i/n)  # smooth


    """

    name: str
    procedure: ProcedureType
    control: str = "force"
    increments: int = 1  # number of load steps
    ramp: None | object = None  # user lambda or None
    compute_reactions: bool = False
    output_fields: list[str] | None = None
    output_frequency: int = 0  # 0 = no dump
    output_save_to: str | None = None  # target folder
    output_file: str | None = None
    verbose: bool = True

    last_solver: object | None = None

    # Store solver reference for post-processing
    _last_solver: LinearStaticSolver | None = field(
        default=None, init=False, repr=False
    )

    def execute(
        self,
        model: Model,
        model_state: ModelState | None = None,
        use_sparse: bool = True,
    ) -> ModelState:
        """
        Execute this analysis step using the current state of the Model.

        This method performs the analysis procedure specified by the Step
        (e.g., a linear static solve), generates a Solution, and stores it
        in the provided ModelState. If no ModelState is supplied, a new one
        is created.

        Boundary conditions and loads in the Model are not modified or
        reset by this method. Each step uses whatever BCs and loading are
        currently defined on the Model. To run different load cases in
        successive steps, the user must update `model.bc` between calls.

        Args:
            model: The model containing the mesh, boundary conditions,
                material properties, and DOF definitions.
            model_state: An existing ModelState used to record solutions
                        across multiple steps. If None, a new ModelState
                        is initialized.
            use_sparse: Whether to use the sparse solver backend
                        (default: True).

        Returns:
            The updated ModelState containing the solution from this step.

        Raises:
            NotImplementedError: If the specified procedure type is not
                                implemented.

        Example:
            model_state = ModelState()
            step = Step(name="Analysis1",
                        procedure=ProcedureType.STATIC_LINEAR)
            model_state = step.execute(model, model_state, use_sparse=False)
        """

        # Create state if not provided (first step)
        if model_state is None:
            model_state = ModelState()

        # Print header if verbose
        if self.verbose:
            self._print_header()
            print(model)

        # Execute based on procedure type
        if self.procedure == ProcedureType.STATIC_LINEAR:
            solution = self._execute_static_linear(model, use_sparse)

        elif self.procedure is ProcedureType.STATIC_LINEAR_INCREMENTAL:
            solution = self._execute_static_linear_incremental(
                model, model_state, use_sparse
            )

        else:
            raise NotImplementedError(
                f"Procedure {self.procedure.name} not yet implemented. "
                f"Currently supported: {[p.name for p in ProcedureType]}"
            )

        # Update model state with this step's solution
        model_state.update_solution(self.name, solution)

        # Store solution also in the Model (used for export/postprocessing)
        model.set_solution(solution)

        # Print completion if verbose
        if self.verbose:
            self._print_completion(solution)

        return model_state

    def _execute_static_linear(self, model: Model, use_sparse: bool) -> Solution:
        """
        Execute a linear static analysis using the current state of the Model.

        The workflow is:
            1. Create the linear static solver.
            2. Assemble the global stiffness matrix.
            3. Apply the boundary conditions and external forces defined
               on the Model at the time of execution.
            4. Solve the linear system K U = F (with static condensation
            if needed).
            5. Compute reaction forces.

        This method performs a fresh linear solve; no information from
        previous steps is reused unless the Model itself has been modified
        by the user.

        Args:
            model: The Model to analyze.
            use_sparse: Whether to use the sparse linear solver backend.

        Returns:
            A Solution object containing displacements, reactions,
            and other solver results for this step.
        """
        # Create solver
        solver = LinearStaticSolver(model, use_sparse=use_sparse)

        # Standard FEA steps
        solver.assemble_global_matrix()
        solver.apply_boundary_conditions()
        solution = solver.solve(compute_reactions=True)

        # Store solver for potential post-processing
        self._last_solver = solver

        return solution

    def _execute_static_linear_incremental(self, model, model_state, use_sparse=True):
        """
        Execute an incremental linear static analysis.

        Supports three control modes:
        - "force":        only Neumann loads are scaled by λ.
        - "displacement": only prescribed displacements are scaled by λ.
        - "mixed":        both forces and prescribed displacements are scaled by λ.

        The load factor λ for increment i is given by self._load_factor(i).
        """

        last_solution = None

        for i in range(1, self.increments + 1):
            lam = self._load_factor(i)

            if self.verbose:
                print(f"[{self.name}] Increment {i}/{self.increments} (λ={lam:.4f})")

            # Scale boundary conditions according to control mode
            if self.control == "force":
                # Displacements stay at their full, user-defined values.
                # Forces will be scaled inside apply_boundary_conditions(load_factor=lam).
                pass

            elif self.control == "displacement":
                # Scale prescribed displacements by λ, keep forces as given.
                model.bc.scale_displacements(lam)

            elif self.control == "mixed":
                # Scale both displacements and forces by λ.
                model.bc.scale_displacements(lam)
            else:
                raise NotImplementedError(
                    f"Unknown control mode '{self.control}'. "
                    "Supported: 'force', 'displacement', 'mixed'."
                )

            # Create and run solver for this increment
            solver = LinearStaticSolver(model, use_sparse=use_sparse)
            solver.assemble_global_matrix()

            if self.control == "force":
                solver.apply_boundary_conditions(load_factor=lam)
            elif self.control == "displacement":
                # Displacements are already scaled in the registry; do not scale forces.
                solver.apply_boundary_conditions(load_factor=1.0)
            elif self.control == "mixed":
                # Forces scaled by λ, displacements already scaled above.
                solver.apply_boundary_conditions(load_factor=lam)

            last_solution = solver.solve(self.compute_reactions)

            # Store solution in model (for export/postprocessing)
            model.set_solution(last_solution)
            self._last_solver = solver
            self.last_solver = solver

            # Incremental VTK output
            if (
                self.output_file is not None
                and self.output_frequency > 0
                and (i % self.output_frequency == 0)
            ):
                self._write_incremental_output(model, last_solution, i)

        if last_solution is None:
            raise RuntimeError("No solution computed in incremental step.")

        return last_solution

    # Helper for load factor
    def _load_factor(self, i):
        """Return load factor λ(i) for increment i (1-based)."""
        if self.increments <= 0:
            raise ValueError("Step.increments must be >= 1")

        if self.ramp is None:
            return i / self.increments  # default linear ramp

        return self.ramp(i, self.increments)

    def _write_incremental_output(self, model, solution, inc):
        fields = self.output_fields or ["displacements"]

        # Compute required fields using a PostProcessor
        # At the moment, we only support "element_stresses" here.
        needs_stresses = "element_stresses" in fields

        if needs_stresses:
            postprocessor = PostProcessor(
                model=model,
                solution=solution,
                global_stiffness_matrix=(
                    self._last_solver.global_stiffness_matrix
                    if self._last_solver is not None
                    else None
                ),
                magnification_factor=1.0,
            )
            postprocessor.compute_element_stresses()

        # Ensure directory exists
        directory = os.path.dirname(self.output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Build output filename
        base, ext = self.output_file.rsplit(".", 1)
        filename = f"{base}_{inc:03d}.{ext}"

        # Map fields ---> VTKWriter flags
        kwargs = {}
        if "displacements" in fields:
            kwargs["displacements"] = True
        if "element_stresses" in fields:
            kwargs["stresses"] = True

        # TODO: NOT COMPUTED YET: strains and reactions
        # "strains": "element_strains" in fields,
        # "reactions": "reactions" in fields and self.compute_reactions,

        model.set_solution(solution)
        model.save_vtk(
            filename,
            displacements="displacements" in fields,
            stresses="element_stresses" in fields,
            save_to=self.output_save_to,
        )

        if self.verbose:
            print(f"    [VTK dump] {filename}")

    def postprocess(
        self,
        model: Model,
        model_state: ModelState,
        operations: list[str] | None = None,
    ) -> None:
        """
        Run post-processing operations for this step.

        This method retrieves the solution associated with this Step from
        the provided ModelState and executes one or more post-processing
        routines such as strain energy evaluation, mesh visualization, or
        equilibrium checking.

        If the underlying solver is unavailable (for example, if this step
        was not executed in the current session), operations that require
        the global stiffness matrix will issue a warning and be skipped.

        Args:
            model: The model that was analyzed.
            model_state: The ModelState containing the solution for this step.
            operations: A list of post-processing operations to perform.
                Available operations include:
                    - "strain_energy_global": Compute 1/2 U^T K U using the
                    global stiffness matrix.
                    - "strain_energy_local": Sum element-level strain energy
                    contributions.
                    - "undeformed_mesh": Plot the original mesh.
                    - "deformed_mesh": Plot the deformed mesh using the
                    stored displacement field.
                    - "equilibrium_check": Evaluate the residual
                    (K U - F - R) \approx 0 and report whether equilibrium
                    is satisfied.

        Raises:
            KeyError: If the solution for this step is not found in the
                    ModelState.
            ValueError: If an unknown operation name is requested.

        Example:
            step.postprocess(
                model,
                state,
                operations=["strain_energy_local", "deformed_mesh"]
            )
        """
        if operations is None:
            operations = []

        # Get solution for this step
        solution = model_state.get_step_solution(self.name)

        # Create post-processor
        postprocessor = PostProcessor(
            model=model,
            solution=solution,
            global_stiffness_matrix=(
                self._last_solver.global_stiffness_matrix if self._last_solver else None
            ),
            magnification_factor=1.0,
        )

        # Execute requested operations
        for op in operations:
            if op == "strain_energy_global":
                if self._last_solver is None:
                    print(
                        f"Warning: Cannot compute strain_energy_global - "
                        f"solver not available for step '{self.name}'"
                    )
                else:
                    postprocessor.compute_strain_energy_global()

            elif op == "strain_energy_local":
                postprocessor.compute_strain_energy_local()

            elif op == "element_stresses":
                postprocessor.compute_element_stresses()

            elif op == "undeformed_mesh":
                postprocessor.undeformed_mesh()

            elif op == "deformed_mesh":
                postprocessor.deformed_mesh()

            elif op == "equilibrium_check":
                if self._last_solver is None:
                    print(
                        f"Warning: Cannot check equilibrium - "
                        f"solver not available for step '{self.name}'"
                    )
                else:
                    is_ok = solution.check_equilibrium(
                        self._last_solver, tolerance=1e-10, verbose=True
                    )
                    if is_ok:
                        print("Equilibrium satisfied")
                    else:
                        print("Equilibrium NOT satisfied")
            else:
                raise ValueError(
                    f"Unknown post-processing operation: '{op}'. "
                    f"Valid options: strain_energy_global, strain_energy_local, "
                    f"undeformed_mesh, deformed_mesh, equilibrium_check"
                )

    def _print_header(self) -> None:
        """Print step execution header."""
        print("\n" + "=" * 70)
        print(f"EXECUTING STEP: {self.name}")
        print(f"  Procedure: {self.procedure.name}")
        print("=" * 70)

    def _print_completion(self, solution: Solution) -> None:
        """Print step completion summary."""
        print("\n" + "=" * 70)
        print(f"STEP '{self.name}' COMPLETE")
        print("=" * 70)
        # Solution already prints its own statistics via solver
