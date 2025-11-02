"""
Two-spring example

This script performs a finite element analysis of a simple two-spring system.

Functions included:

- assemble_global_stiffness_matrix: Assembles the global stiffness matrix.
- apply_nodal_forces: Applies nodal forces (Neumann boundary conditions).
- apply_prescribed_displacements: Applies prescribed displacements (Dirichlet boundary conditions).
- invert_global_stiffness_matrix: Computes the inverse of the global stiffness matrix.
- solve_nodal_displacements: Solves for the nodal displacements.
- compute_strain_energy_local: Computes the total strain energy through element-wise computation.
- compute_strain_energy_global: Computes the total strain energy using the global solution.
- main: Defines the solution procedure.

"""

import numpy as np


def assemble_global_stiffness_matrix(
    num_elements: int,
    element_stiffness: list[float],
    element_connectivity: list[list[int]],
    global_stiffness_matrix: np.ndarray,
) -> np.ndarray:
    """
    Assembles the global stiffness matrix by integrating element stiffness matrices.

    Returns:
        The fully assembled global stiffness matrix.
    """

    # Assemble the global stiffness matrix
    print("\n- Assembling local stiffness matrix into global stiffness matrix")
    for element_index in range(num_elements):
        msg = (
            f"\n-- Generating stiffness matrix for element {element_index}"
            f" with stiffness {element_stiffness[element_index]}"
        )
        print(msg)

        # Generate the local stiffness matrix for a one-dimensional spring element
        stiffness_value = element_stiffness[element_index]
        local_stiffness_matrix = [
            [stiffness_value, -stiffness_value],
            [-stiffness_value, stiffness_value],
        ]
        print(local_stiffness_matrix)

        # Map local degrees of freedom to global degrees of freedom for an element
        # first determine the element nodes through the element connectivity matrix
        element_nodes = element_connectivity[element_index]
        # then build the local to global DOF mapping
        # For elements with one DOF per node, the global DOF is the same as the node number
        dof_mapping = element_nodes

        # Assemble the local stiffness matrix into the global stiffness matrix
        for i in range(len(dof_mapping)):
            global_i = dof_mapping[i]
            for j in range(len(dof_mapping)):
                global_j = dof_mapping[j]
                global_stiffness_matrix[global_i][global_j] += local_stiffness_matrix[
                    i
                ][j]

    return global_stiffness_matrix


def apply_nodal_forces(
    applied_forces: list[list[float]],
    global_force_vector: np.ndarray,
) -> np.ndarray:
    """
    Applies nodal forces to the global force vector (Neumann boundary conditions).

    Returns:
        The updated global force vector.
    """

    for dof, value in applied_forces:
        global_force_vector[int(dof)] = value

    return global_force_vector


def apply_prescribed_displacements(
    prescribed_displacements: list[list[float]],
    global_stiffness_matrix: np.ndarray,
    global_force_vector: np.ndarray,
    total_dofs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies the prescribed displacements by modifying the global stiffness
    matrix and force vector (Dirichlet boundary conditions).

    Returns:
        A tuple containing the updated global stiffness matrix and global force vector.
    """

    for dof, value in prescribed_displacements:
        for i in range(total_dofs):
            global_stiffness_matrix[i][int(dof)] = 0.0  # Zero out the column
            global_stiffness_matrix[int(dof)][i] = 0.0  # Zero out the row
        global_stiffness_matrix[int(dof)][int(dof)] = 1.0  # Put one in the diagonal
        global_force_vector[int(dof)] = 0.0

    return (
        global_stiffness_matrix,
        global_force_vector,
    )


def invert_global_stiffness_matrix(
    global_stiffness_matrix: np.ndarray,
) -> list[list[float]]:
    """
    Computes the inverse of the global stiffness matrix using the Gauss-Jordan elimination method.

    Returns:
        The inverse of the global stiffness matrix.
    """

    # Create an identity matrix (this will store the inverse at the end of the process)
    n = len(global_stiffness_matrix)
    inverse_global_stiffness_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(0.0)
        inverse_global_stiffness_matrix.append(row)

    epsilon = 1e-12  # Tolerance for zero check

    # Perform Gauss-Jordan elimination
    for i in range(n):
        # Check for zero on the diagonal
        diag_factor = global_stiffness_matrix[i][i]
        if abs(diag_factor) < epsilon:
            raise ValueError(
                "Matrix is singular or nearly singular and cannot be inverted."
            )
        # Make the diagonal contain all ones
        for j in range(n):
            global_stiffness_matrix[i][j] /= diag_factor
            inverse_global_stiffness_matrix[i][j] /= diag_factor
        # Make the other elements in column i be zeros
        for k in range(n):
            if k != i:
                factor = global_stiffness_matrix[k][i]
                for j in range(n):
                    global_stiffness_matrix[k][j] -= (
                        factor * global_stiffness_matrix[i][j]
                    )
                    inverse_global_stiffness_matrix[k][j] -= (
                        factor * inverse_global_stiffness_matrix[i][j]
                    )

    return inverse_global_stiffness_matrix


def solve_nodal_displacements(
    total_dofs: int,
    inverse_global_stiffness_matrix: list[list[float]],
    global_force_vector: list[float],
) -> list[float]:
    """
    Solves for the nodal displacements.

    Returns:
        The calculated nodal displacements.
    """

    nodal_displacements = []
    for i in range(total_dofs):
        displacement = 0.0
        for j in range(total_dofs):
            displacement += (
                inverse_global_stiffness_matrix[i][j] * global_force_vector[j]
            )
        nodal_displacements.append(displacement)

    return nodal_displacements


def compute_strain_energy_local(
    num_elements: int,
    element_stiffness: list[float],
    element_connectivity: list[list[int]],
    nodal_displacements: list[float],
) -> None:
    """
    Computes the total strain energy by summing the strain energy of each element.

    Returns:
        None.
    """

    total_strain_energy = 0.0
    for element_index in range(num_elements):
        k_spring = element_stiffness[element_index]
        node1, node2 = element_connectivity[element_index]
        u1 = nodal_displacements[node1]
        u2 = nodal_displacements[node2]
        delta = u2 - u1
        strain_energy = 0.5 * k_spring * delta**2
        total_strain_energy += strain_energy
        print(f"\n- Strain energy in element {element_index}: {strain_energy}")
    print(
        f"\n- Total strain energy in the system (from local computation): {total_strain_energy}"
    )


def compute_strain_energy_global(
    total_dofs: int,
    original_global_stiffness_matrix: list[list[float]],
    nodal_displacements: list[float],
) -> None:
    """
    Computes the total strain energy using the global solution (U = 0.5 * u^T * K * u).

    Returns:
        None.
    """

    total_strain_energy = 0.0

    # Compute K * u
    K_u = []
    for i in range(total_dofs):
        value = 0.0
        for j in range(total_dofs):
            value += original_global_stiffness_matrix[i][j] * nodal_displacements[j]
        K_u.append(value)

    # Compute u^T * (K * u)
    for i in range(total_dofs):
        total_strain_energy += nodal_displacements[i] * K_u[i]

    # Multiply by 0.5
    total_strain_energy *= 0.5

    print(
        f"\n- Total strain energy in the system (from global computation): {total_strain_energy}"
    )


def main() -> None:
    # Preprocessing

    # - Define input data
    num_nodes = 3
    dofs_per_node = 1
    num_elements = 2
    nodes_per_element = 2

    # - Define discretization

    # -- Connectivity matrix defining which nodes belong to each element
    element_connectivity = [
        [1, 2],
        [2, 0],
    ]

    # - Define material properties

    # -- Stiffness properties for each spring element
    element_stiffness = [1.0, 2.0]

    # - Define boundary conditions

    # -- Prescribed displacements (Dirichlet boundary conditions): [DOF, value]
    prescribed_displacements = [
        [1, 0.0],  # DOF 0 is constrained
    ]

    # -- Applied forces (Neumann boundary conditions): [DOF, value]
    applied_forces = [
        [0, 10.0],  # DOF 1 has an applied force of 10
    ]

    # - Initialize arrays

    # -- Compute total number of DOFs
    total_dofs = dofs_per_node * num_nodes

    # -- Initialize the global stiffness matrix as a square matrix of zeros
    global_stiffness_matrix = np.zeros((total_dofs, total_dofs))

    # -- Initialize the global force vector with zeros
    global_force_vector = np.zeros((0, total_dofs))

    # Processing

    # - Assemble the global stiffness matrix
    global_stiffness_matrix = assemble_global_stiffness_matrix(
        num_elements, element_stiffness, element_connectivity, global_stiffness_matrix
    )
    print("\n- Global stiffness matrix K:")
    for row in global_stiffness_matrix:
        print(row)

    # - Save a copy of the original global stiffness matrix before applying boundary conditions
    original_global_stiffness_matrix = []
    for row in global_stiffness_matrix:
        original_global_stiffness_matrix.append(row[:])

    # - Boundary conditions: Apply forces
    global_force_vector = apply_nodal_forces(applied_forces, global_force_vector)

    # - Boundary conditions: Constrain displacements
    global_stiffness_matrix, global_force_vector = apply_prescribed_displacements(
        prescribed_displacements,
        global_stiffness_matrix,
        global_force_vector,
        total_dofs,
    )

    print("\n- Modified global stiffness matrix K after applying boundary conditions:")
    for row in global_stiffness_matrix:
        print(row)

    print("\n- Global force vector F after applying boundary conditions:")
    print(global_force_vector)

    # - Compute the inverse of the global stiffness matrix using the Gauss-Jordan Elimination method
    inverse_global_stiffness_matrix = invert_global_stiffness_matrix(
        global_stiffness_matrix
    )

    # - Solve for the nodal displacements
    nodal_displacements = solve_nodal_displacements(
        total_dofs, inverse_global_stiffness_matrix, global_force_vector
    )
    print("\n- Nodal displacements U:")
    print(nodal_displacements)

    # Postprocessing: Calculate strain energy for each spring and for system of springs

    # - Compute strain energy at element level
    compute_strain_energy_local(
        num_elements, element_stiffness, element_connectivity, nodal_displacements
    )

    # - Compute strain energy at system level
    compute_strain_energy_global(
        total_dofs, original_global_stiffness_matrix, nodal_displacements
    )


if __name__ == "__main__":
    main()
