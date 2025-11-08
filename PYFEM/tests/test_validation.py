#!/usr/bin/env python
"""
Test comprehensive model validation.

Created: 2025/10/29 19:27:44
Last modified: 2025/11/08 18:13:41
Author: Angelo Simone (angelo.simone@unipd.it)
"""

import numpy as np

import pyfem


def test_valid_model():
    """Test 1: Valid model - should pass all validations."""
    print("=" * 70)
    print("Test 1: Valid 1D model with compatible elements")
    print("=" * 70)

    # Create mesh
    mesh = pyfem.Mesh(
        num_nodes=3,
        points=np.array([0.0, 1.0, 2.0]),
        num_elements=2,
        element_connectivity=[[0, 1], [1, 2]],
        element_property_labels=["bar", "spring"],
    )

    # Define compatible 1D elements
    element_properties = pyfem.make_element_properties(
        [
            ("bar", ("bar_1D", {"E": 200e9, "A": 0.01})),
            ("spring", ("spring_1D", {"k": 1000.0})),
        ]
    )

    # Create model
    problem = pyfem.Problem(pyfem.Physics.MECHANICS, pyfem.Dimension.D1)
    model = pyfem.Model(mesh, problem)

    try:
        model.set_element_properties(element_properties)
        print("SUCCESS: All validations passed")
        print("   - Mesh validation: OK")
        print("   - Element-problem compatibility: OK")
        print("   - Elements: bar_1D, spring_1D")
        print(f"   - Problem: {problem}")
    except pyfem.ModelValidationError as e:
        print(f"FAILED: {e}")

    print()


def test_incompatible_elements():
    """Test 2: Incompatible elements - should fail validation."""
    print("=" * 70)
    print("Test 2: 1D problem with 2D element (should fail)")
    print("=" * 70)

    # Create mesh
    mesh = pyfem.Mesh(
        num_nodes=3,
        points=np.array([0.0, 1.0, 2.0]),
        num_elements=2,
        element_connectivity=[[0, 1], [1, 2]],
        element_property_labels=["bar", "quad"],
    )

    # Define incompatible elements (quad4 is 2D, but problem is 1D)
    element_properties = pyfem.make_element_properties(
        [
            ("bar", ("bar_1D", {"E": 200e9, "A": 0.01})),
            ("quad", ("quad4", {"E": 200e9, "nu": 0.3})),
        ]
    )

    # Create model
    problem = pyfem.Problem(pyfem.Physics.MECHANICS, pyfem.Dimension.D1)
    model = pyfem.Model(mesh, problem)

    try:
        model.set_element_properties(element_properties)
        print("FAILED: Should have caught incompatible element!")
    except pyfem.ModelValidationError as e:
        print("SUCCESS: Validation caught incompatibility")
        print(f"   Error: {e}")

    print()


def test_missing_parameters():
    """Test 3: Missing required parameters - should fail validation."""
    print("=" * 70)
    print("Test 3: Element missing required parameters (should fail)")
    print("=" * 70)

    # Create mesh
    mesh = pyfem.Mesh(
        num_nodes=3,
        points=np.array([0.0, 1.0, 2.0]),
        num_elements=2,
        element_connectivity=[[0, 1], [1, 2]],
        element_property_labels=["bar", "bar"],
    )

    # Define element with missing parameter (A is missing)
    element_properties = pyfem.make_element_properties(
        [
            ("bar", ("bar_1D", {"E": 200e9})),  # Missing "A" parameter!
        ]
    )

    # Create model
    problem = pyfem.Problem(pyfem.Physics.MECHANICS, pyfem.Dimension.D1)
    model = pyfem.Model(mesh, problem)

    try:
        model.set_element_properties(element_properties)
        print("FAILED: Should have caught missing parameter!")
    except pyfem.ModelValidationError as e:
        print("SUCCESS: Validation caught missing parameter")
        print(f"   Error: {e}")

    print()


def test_query_compatibility():
    """Test 4: Query element compatibility."""
    print("=" * 70)
    print("Test 4: Query element compatibility")
    print("=" * 70)

    # Check what problem configurations bar_1D can be used with
    compatible = pyfem.get_compatible_problems("bar_1D")
    print("bar_1D is compatible with:")
    for physics, dimension in compatible:
        print(f"  - {physics.name}, {dimension.value}D")

    print()

    # Check what problem configurations quad4 can be used with
    compatible = pyfem.get_compatible_problems("quad4")
    print("quad4 is compatible with:")
    for physics, dimension in compatible:
        print(f"  - {physics.name}, {dimension.value}D")

    print()

    # Check specific compatibility
    problem_1d = pyfem.Problem(pyfem.Physics.MECHANICS, pyfem.Dimension.D1)
    problem_2d = pyfem.Problem(pyfem.Physics.MECHANICS, pyfem.Dimension.D2)

    is_compat = pyfem.is_element_compatible_with_problem("bar_1D", problem_1d)
    print(f"Is bar_1D compatible with 1D mechanics? {is_compat}")

    is_compat = pyfem.is_element_compatible_with_problem("bar_1D", problem_2d)
    print(f"Is bar_1D compatible with 2D mechanics? {is_compat}")

    is_compat = pyfem.is_element_compatible_with_problem("quad4", problem_2d)
    print(f"Is quad4 compatible with 2D mechanics? {is_compat}")

    print()


def main():
    """Run all validation tests."""
    test_valid_model()
    test_incompatible_elements()
    test_missing_parameters()
    test_query_compatibility()

    print("=" * 70)
    print("All validation tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
