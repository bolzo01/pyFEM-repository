#!/usr/bin/env python
"""
Element property registry.


This module defines how element properties are specified and validated
in the finite element model. An ElementProperty describes:

    - kind:   the element type (e.g. "spring_1D", "bar_1D")
    - params: the required numerical parameters for that element
    - meta:   optional discretization metadata (e.g. integration order)
    - material: the material label assigned to the element

Only the new format is supported:

    ElementProperty(
        kind="bar_1D",
        params={"A": 7.0},
        material="steel",
        meta={"integration": 3}
    )

The registry is built by calling:

    element_properties = make_element_properties([
        ("bar", ElementProperty(...)),
        ("spring", ElementProperty(...)),
    ])

Created: 2025/10/19 00:16:39
Last modified: 2025/11/17 21:50:44
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ElementProperty:
    """
    Container describing the properties of a single finite element type
    used in the model: kind, numerical parameters, discretization metadata,
    and material assignment.
    """

    kind: str
    params: dict[str, object]
    meta: dict[str, object] = field(default_factory=dict)
    material: str | None = None


# Registry type alias
ElementProperties = dict[str, ElementProperty]


class ElementPropertyError(RuntimeError):
    """Raised on missing or invalid element property definitions."""

    pass


def param(elem_prop: ElementProperty, key: str, cast: type | None = None):
    """Fetch parameter by name."""
    if key not in elem_prop.params:
        raise ElementPropertyError(
            f"Parameter '{key}' not found in element of kind '{elem_prop.kind}'."
        )
    val = elem_prop.params[key]
    return cast(val) if cast is not None else val


def make_element_properties(pairs) -> ElementProperties:
    """
    Build an element properties registry from a list of (label, ElementProperty) pairs.

    Input format:
        [
            ("bar", ElementProperty(
                kind="bar_1D",
                params={"A": 7.0},
                material="steel",
                meta={"integration": 3, "supplier": "ACME Corp"},
            )),
        ]

    Only ElementProperty instances are accepted. Older tuple formats
    (e.g., ("bar_1D", {...})) are no longer supported.

    Returns:
        dict[str, ElementProperty]
    """

    # Ensure it's iterable
    try:
        iterator = iter(pairs)
    except TypeError as exc:
        raise ValueError(
            "make_element_properties expects a list/iterable of (label, entry) pairs."
        ) from exc

    out: ElementProperties = {}
    seen: set[str] = set()

    for idx, pair in enumerate(iterator):
        # Basic shape check
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise ValueError(
                f"Item #{idx} must be a (label, entry) pair, got: {pair!r}"
            )

        label, entry = pair

        # Label checks
        if not isinstance(label, str) or not label:
            raise ValueError(f"Item #{idx} has invalid label: {label!r}")

        if label in seen:
            raise ElementPropertyError(
                f"Duplicate element property label '{label}' in specification (at item #{idx})."
            )
        seen.add(label)

        # Normalize entry to ElementProperty
        if not isinstance(entry, ElementProperty):
            raise ValueError(
                f"Item #{idx}: entry must be an ElementProperty instance.\n"
                f"Use: ElementProperty(kind=..., params=..., material=..., meta=...).\n"
                f"Got: {entry!r}"
            )

        elem_prop = entry

        out[label] = elem_prop

    return out


# Validation

REQUIRED_PARAMS: dict[str, set[str]] = {
    "spring_1D": {"k"},
    "bar_1D": {"A"},
    "bar3_1D": {"A"},
    "bar_2D": {"A"},
    # "beam_1D": {"A", "I"},
}

type MetaSpec = set[str | int]

ALLOWED_META: dict[str, dict[str, MetaSpec]] = {
    "spring_1D": {},
    "bar_1D": {
        "interpolation": {"linear", "quadratic", "cubic"},
        "integration": {"analytical", "full", "reduced", 1, 2, 3},
    },
    # "beam_1D": {
    #     "interpolation": {"linear", "cubic"},
    #     "integration": {"analytical", "full", "reduced"},
    # },
    # "plane_stress": {
    #     "interpolation": {"linear", "quadratic"},
    #     "integration": {"full", "reduced"},
    #     "formulation": {"standard", "enhanced"},
    # },
}


def validate_mesh_and_element_properties(
    mesh, element_properties: ElementProperties
) -> None:
    """
    Validate that:
      1) element kinds are known (declared in REQUIRED_PARAMS);
      2) each element has the required parameters for its kind;
      3) meta fields contain valid values (only those declared in ALLOWED_META);
      4) the mesh provides one element property label per element; and
      5) every label used by the mesh exists in the registry.
    """
    # 1, 2) Registry: kinds must be known + required params present
    allowed_kinds = set(REQUIRED_PARAMS.keys())
    for label, elem_prop in element_properties.items():
        if elem_prop.kind not in allowed_kinds:
            raise ElementPropertyError(
                f"Element property '{label}' uses unknown kind '{elem_prop.kind}'. "
                f"Allowed kinds: {sorted(allowed_kinds)}. "
                "Add the kind to REQUIRED_PARAMS if you intend to use it."
            )
        missing = REQUIRED_PARAMS[elem_prop.kind] - elem_prop.params.keys()
        if missing:
            raise ElementPropertyError(
                f"Element property '{label}' (kind={elem_prop.kind}) is missing required "
                f"parameter(s): {', '.join(sorted(missing))}"
            )

        # 3) Validate meta fields (only those declared in ALLOWED_META)
        if elem_prop.kind in ALLOWED_META:
            allowed_meta_for_kind = ALLOWED_META[elem_prop.kind]

            # Reject accidental use of meta["material"]
            if "material" in elem_prop.meta:
                raise ElementPropertyError(
                    f"Element property '{label}' includes 'material' inside meta.\n"
                    f"Material must be specified using ElementProperty(material=...)."
                )

            for meta_key, meta_value in elem_prop.meta.items():
                # Only validate keys explicitly listed in ALLOWED_META
                if meta_key in allowed_meta_for_kind:
                    allowed_values = allowed_meta_for_kind[meta_key]
                    if meta_value not in allowed_values:
                        raise ElementPropertyError(
                            f"Element property '{label}' meta field '{meta_key}' has invalid value '{meta_value}'. "
                            f"Allowed values: {allowed_values}"
                        )
                # Otherwise: meta_key is informational — no validation required

    # 4) Mesh assignment length
    labels: list[str] = mesh.element_property_labels
    if mesh.num_elements != len(labels):
        raise ValueError(
            "mesh.element_property_labels must contain one label per element "
            f"(got {len(labels)} for {mesh.num_elements} elements)."
        )

    # 5) Every used label exists in the registry
    used = set(labels)
    known = set(element_properties.keys())
    unknown = sorted(used - known)
    if unknown:
        raise ElementPropertyError(
            "Unknown element property label(s) used by the mesh: " + ", ".join(unknown)
        )

    return None
