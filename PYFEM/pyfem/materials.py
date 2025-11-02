"""
Material registry.

Created: 2025/09/04 11:52:21
Last modified: 2025/11/02 19:19:17
Author: Angelo Simone (angelo.simone@unipd.it)
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Material:
    """Constitutive model and parameters.

    kind   : identifier of the constitutive model / element law
             (e.g., "spring_1D", "plane_stress", "neo_hookean").
    params : mapping of required parameters for that model
             (e.g., {"k": 2.0} or {"E": 210e9, "nu": 0.3}).
    meta   : optional free-form notes/units/provenance (ignored by solvers).
    """

    kind: str
    params: dict[str, object]
    meta: dict[str, object] = field(default_factory=dict)


# Registry type alias
Materials = dict[str, Material]


class MaterialError(RuntimeError):
    """Raised on missing or invalid material definitions."""

    pass


def param(mat: Material, key: str, cast: type | None = None):
    """Fetch parameter by name."""
    if key not in mat.params:
        raise MaterialError(
            f"Parameter '{key}' not found in material of kind '{mat.kind}'."
        )
    val = mat.params[key]
    return cast(val) if cast is not None else val


def make_materials(pairs) -> dict[str, Material]:
    """
    Build a materials registry from a list of (label, entry) pairs.

    Input format:
        [
            ("soft",  ("spring_1D", {"k": 1.0})),
            ("stiff", ("spring_1D", {"k": 2.0})),
            ("al_2024", ("plane_stress",{"E": 73e9, "nu": 0.33})),
            ("soft", ("spring_1D", {"k": 0.5})),  # this would raise MaterialError
        ]

    Each entry can be:
        - Material(kind, params[, meta])
        - (kind: str, params: dict[str, object])
        - (kind: str, params: dict[str, object], meta: dict[str, object])

    Returns:
        dict[str, Material]
    """
    # Ensure it's iterable
    try:
        iterator = iter(pairs)
    except TypeError as exc:
        raise ValueError(
            "make_materials expects a list/iterable of (label, entry) pairs."
        ) from exc

    out: dict[str, Material] = {}
    seen: set[str] = set()

    for idx, pair in enumerate(iterator):
        # basic shape check
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise ValueError(
                f"Item #{idx} must be a (label, entry) pair, got: {pair!r}"
            )

        label, entry = pair

        # label checks
        if not isinstance(label, str) or not label:
            raise ValueError(f"Item #{idx} has invalid label: {label!r}")

        if label in seen:
            raise MaterialError(
                f"Duplicate material label '{label}' in specification (at item #{idx})."
            )
        seen.add(label)

        # normalize entry: Material
        if isinstance(entry, Material):
            mat = entry
        else:
            if not isinstance(entry, (tuple, list)) or len(entry) not in (2, 3):
                raise ValueError(
                    f"Item #{idx} entry must be Material or (kind, params[, meta]), got: {entry!r}"
                )
            kind = entry[0]
            params = entry[1]
            meta = entry[2] if len(entry) == 3 else {}

            if not isinstance(kind, str) or not kind:
                raise ValueError(f"Item #{idx} has invalid kind: {kind!r}")
            if not isinstance(params, dict):
                # allow dict-like mappings but store a real dict
                try:
                    params = dict(params)
                except Exception as exc:
                    raise ValueError(
                        f"Item #{idx} params must be a mapping, got: {type(params)!r}"
                    ) from exc
            if not isinstance(meta, dict):
                try:
                    meta = dict(meta)
                except Exception as exc:
                    raise ValueError(
                        f"Item #{idx} meta must be a mapping, got: {type(meta)!r}"
                    ) from exc

            mat = Material(kind=kind, params=dict(params), meta=dict(meta))

        out[label] = mat

    return out


# Validation

REQUIRED_PARAMS: dict[str, set[str]] = {
    "spring_1D": {"k"},
    "bar_1D": {"E", "A"},
    # "plane_strain": {"E", "nu"},
}


def validate_mesh_and_materials(mesh, materials: Materials) -> None:
    """
    Validate that:
      1) material kinds are known (declared in REQUIRED_PARAMS);
      2) each material has the required parameters for its kind;
      3) the mesh provides one material label per element; and
      4) every label used by the mesh exists in the registry.
    """
    # 1, 2) Registry: kinds must be known + required params present
    allowed_kinds = set(REQUIRED_PARAMS.keys())
    for label, mat in materials.items():
        if mat.kind not in allowed_kinds:
            raise MaterialError(
                f"Material '{label}' uses unknown kind '{mat.kind}'. "
                f"Allowed kinds: {sorted(allowed_kinds)}. "
                "Add the kind to REQUIRED_PARAMS if you intend to use it."
            )
        missing = REQUIRED_PARAMS[mat.kind] - mat.params.keys()
        if missing:
            raise MaterialError(
                f"Material '{label}' (kind={mat.kind}) is missing required "
                f"parameter(s): {', '.join(sorted(missing))}"
            )

    # 3) Mesh assignment length
    labels: list[str] = mesh.element_material
    if mesh.num_elements != len(labels):
        raise ValueError(
            "mesh.element_material must contain one label per element "
            f"(got {len(labels)} for {mesh.num_elements} elements)."
        )

    # 4) Every used label exists in the registry
    used = set(labels)
    known = set(materials.keys())
    unknown = sorted(used - known)
    if unknown:
        raise MaterialError(
            "Unknown material label(s) used by the mesh: " + ", ".join(unknown)
        )

    return None
