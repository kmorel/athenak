#!/usr/bin/env python3
"""Write distributed AthenaK mesh and particle tracks as ADIOS2/Fides data."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np


MHD_VECTORS = {
    "fluid_velocity": ("velx", "vely", "velz"),
    "magnetic_field": ("bcc1", "bcc2", "bcc3"),
}
TRACK_VECTORS = {
    "particle_velocity": ("vx", "vy", "vz"),
    "magnetic_field": ("bx", "by", "bz"),
    "curvature": ("k1", "k2", "k3"),
    "magnetic_field_gradient": ("db1", "db2", "db3"),
}
COORDINATE_FIELDS = ("x", "y", "z")


def output_paths(output: str | Path) -> dict[str, Path]:
    """Return the two BP and two Fides paths associated with an output prefix."""

    path = Path(output)
    if path.suffix.lower() in (".bp", ".json"):
        path = path.with_suffix("")
    return {
        "mesh_bp": path.parent / f"{path.name}.mesh.bp",
        "mesh_json": path.parent / f"{path.name}.mesh.json",
        "tracks_bp": path.parent / f"{path.name}.tracks.bp",
        "tracks_json": path.parent / f"{path.name}.tracks.json",
    }


def visualization_track_fields(fields: Sequence[str] | None) -> list[str] | None:
    """Ensure a requested track-field subset contains track coordinates."""

    if fields is None:
        return None
    return list(dict.fromkeys((*COORDINATE_FIELDS, *fields)))


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _array_groups(
    fields: Sequence[str], definitions: Mapping[str, tuple[str, str, str]]
) -> tuple[dict[str, tuple[str, str, str]], list[str]]:
    available = set(fields)
    vectors = {
        name: components
        for name, components in definitions.items()
        if set(components) <= available
    }
    consumed = {item for components in vectors.values() for item in components}
    scalars = [name for name in fields if name not in consumed and name not in vectors]
    return vectors, scalars


def mesh_schema(block: dict) -> dict:
    """Return the small, rank-independent schema needed to pack mesh arrays."""

    fields = tuple(block["VariableNames"])
    vectors, scalars = _array_groups(fields, MHD_VECTORS)
    return {
        "fields": fields,
        "vectors": vectors,
        "scalars": scalars,
        "dtype": np.asarray(block[fields[0]]).dtype.str,
        "coordinate_dtype": np.asarray(block["x1f"]).dtype.str,
    }


def _block_points(block: dict) -> np.ndarray:
    z, y, x = np.meshgrid(
        np.asarray(block["x3f"]),
        np.asarray(block["x2f"]),
        np.asarray(block["x1f"]),
        indexing="ij",
    )
    return np.ascontiguousarray(np.stack((x, y, z), axis=-1).reshape(-1, 3))


def _block_connectivity(shape: Sequence[int], point_offset: int) -> np.ndarray:
    nz, ny, nx = (int(value) for value in shape)
    plane = (ny + 1) * (nx + 1)
    row = nx + 1
    iz, iy, ix = np.meshgrid(
        np.arange(nz, dtype=np.int64),
        np.arange(ny, dtype=np.int64),
        np.arange(nx, dtype=np.int64),
        indexing="ij",
    )
    base = (iz * plane + iy * row + ix).reshape(-1) + point_offset
    return np.ascontiguousarray(
        np.column_stack(
            (
                base,
                base + 1,
                base + row + 1,
                base + row,
                base + plane,
                base + plane + 1,
                base + plane + row + 1,
                base + plane + row,
            )
        )
    )


def prepare_mesh_arrays(meshblocks: Sequence[dict], schema: dict) -> dict[str, np.ndarray]:
    """Pack local MeshBlocks into an explicit hexahedral mesh."""

    dtype = np.dtype(schema["dtype"])
    points_parts = []
    connectivity_parts = []
    field_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in (*schema["scalars"], *schema["vectors"])
    }
    point_offset = 0
    for block in meshblocks:
        fields = tuple(block["VariableNames"])
        if fields != tuple(schema["fields"]):
            raise ValueError("all MeshBlocks must contain the same fields in the same order")
        shape = np.asarray(block[fields[0]]).shape
        points = _block_points(block)
        points_parts.append(points)
        connectivity_parts.append(_block_connectivity(shape, point_offset))
        point_offset += points.shape[0]
        for name in schema["scalars"]:
            field_parts[name].append(np.asarray(block[name]).reshape(-1))
        for name, components in schema["vectors"].items():
            field_parts[name].append(
                np.stack([np.asarray(block[item]) for item in components], axis=-1)
                .reshape(-1, 3)
            )

    coordinate_dtype = np.dtype(schema["coordinate_dtype"])
    arrays = {
        "points": (
            np.concatenate(points_parts)
            if points_parts
            else np.empty((0, 3), coordinate_dtype)
        ),
        "connectivity": (
            np.concatenate(connectivity_parts).reshape(-1)
            if connectivity_parts
            else np.empty((0,), dtype=np.int64)
        ),
    }
    for name in schema["scalars"]:
        parts = field_parts[name]
        arrays[f"cell_data/{_safe_name(name)}"] = (
            np.concatenate(parts) if parts else np.empty((0,), dtype=dtype)
        )
    for name in schema["vectors"]:
        parts = field_parts[name]
        arrays[f"cell_data/{_safe_name(name)}"] = (
            np.concatenate(parts) if parts else np.empty((0, 3), dtype=dtype)
        )
    return {name: np.ascontiguousarray(value) for name, value in arrays.items()}


def _track_schema(tracks: dict) -> dict:
    fields = tuple(tracks["fields"])
    field = {name: index for index, name in enumerate(fields)}
    missing = sorted(set(COORDINATE_FIELDS) - set(field))
    if missing:
        raise ValueError(f"track geometry fields are missing: {missing}")
    vectors, scalars = _array_groups(fields, TRACK_VECTORS)
    scalars = [
        name for name in scalars
        if name not in COORDINATE_FIELDS and name not in ("time", "cycle")
    ]
    derived = []
    if {"bx", "by", "bz"} <= set(field) and "magnetic_field_magnitude" not in field:
        derived.append("magnetic_field_magnitude")
    if {"k1", "k2", "k3"} <= set(field) and "curvature_magnitude" not in field:
        derived.append("curvature_magnitude")
    if {"vx", "vy", "vz", "bx", "by", "bz"} <= set(field) and "mu_M" not in field:
        derived.append("mu_M")
    return {
        "fields": fields,
        "field": field,
        "vectors": vectors,
        "scalars": scalars,
        "derived": derived,
    }


def prepare_track_arrays(
    tracks: dict,
    domain_bounds: np.ndarray | None = None,
    periodic_axes: Sequence[int] = (),
) -> dict[str, np.ndarray]:
    """Pack local trajectories as points and two-vertex line cells."""

    values = np.asarray(tracks["values"])
    nparticles, ntimes, _ = values.shape
    schema = _track_schema(tracks)
    field = schema["field"]
    coordinates = [field[name] for name in COORDINATE_FIELDS]
    points = np.ascontiguousarray(values[..., coordinates].reshape(-1, 3))

    valid = np.ones((nparticles, max(ntimes - 1, 0)), dtype=bool)
    if valid.size and periodic_axes and domain_bounds is not None:
        axes = np.asarray(periodic_axes, dtype=int)
        bounds = np.asarray(domain_bounds)
        widths = bounds[axes, 1] - bounds[axes, 0]
        valid &= ~np.any(
            np.abs(np.diff(values[..., coordinates], axis=1)[..., axes])
            > 0.5 * widths,
            axis=2,
        )
    particle_index, time_index = np.nonzero(valid)
    first = particle_index * ntimes + time_index
    connectivity = np.column_stack((first, first + 1)).astype(
        np.int64, copy=False
    ).reshape(-1)

    flat = values.reshape(-1, values.shape[-1])
    times = np.broadcast_to(np.asarray(tracks["times"]), (nparticles, ntimes)).reshape(-1)
    cycles = np.broadcast_to(np.asarray(tracks["cycles"]), (nparticles, ntimes)).reshape(-1)
    arrays: dict[str, np.ndarray] = {
        "points": points,
        "connectivity": connectivity,
        "point_data/time": times,
        "point_data/cycle": cycles,
    }
    for name in schema["scalars"]:
        arrays[f"point_data/{_safe_name(name)}"] = flat[:, field[name]]
    for name, components in schema["vectors"].items():
        arrays[f"point_data/{_safe_name(name)}"] = flat[
            :, [field[item] for item in components]
        ]

    if "magnetic_field_magnitude" in schema["derived"] or "mu_M" in schema["derived"]:
        magnetic = flat[:, [field[name] for name in ("bx", "by", "bz")]]
        bmag = np.linalg.norm(magnetic, axis=1)
        if "magnetic_field_magnitude" in schema["derived"]:
            arrays["point_data/magnetic_field_magnitude"] = bmag
    if "curvature_magnitude" in schema["derived"]:
        curvature = flat[:, [field[name] for name in ("k1", "k2", "k3")]]
        arrays["point_data/curvature_magnitude"] = np.linalg.norm(curvature, axis=1)
    if "mu_M" in schema["derived"]:
        velocity = flat[:, [field[name] for name in ("vx", "vy", "vz")]]
        floor = np.array(1.0e-30, dtype=flat.dtype)
        safe_bmag = np.maximum(bmag, floor)
        parallel = np.sum(velocity * magnetic, axis=1) / safe_bmag
        perpendicular2 = np.maximum(
            np.sum(velocity * velocity, axis=1) - parallel * parallel, floor
        )
        arrays["point_data/mu_M"] = perpendicular2 / (2.0 * safe_bmag)
    if "inside_meshblock" in tracks:
        arrays["point_data/inside_meshblock"] = np.asarray(
            tracks["inside_meshblock"], dtype=np.uint8
        ).reshape(-1)

    source_rows = np.asarray(tracks["source_rows"])
    arrays["cell_data/source_row"] = source_rows[particle_index]
    particles = np.asarray(tracks["particles"])
    for name in particles.dtype.names or ():
        arrays[f"cell_data/{_safe_name(name)}"] = particles[name][particle_index]
    return {name: np.ascontiguousarray(value) for name, value in arrays.items()}


def write_bp(
    filename: str | Path,
    arrays: Mapping[str, np.ndarray],
    comm,
    attributes: Mapping[str, str] | None = None,
    array_blocks: Iterable[Mapping[str, np.ndarray]] | None = None,
) -> dict[str, int]:
    """Collectively write local-array blocks for Fides to one BP file.

    ``arrays`` supplies the variable schema for this rank. By default it is
    also the single block written by the rank. ``array_blocks`` may instead
    yield multiple same-shaped blocks, allowing MeshBlocks to be packed and
    released one at a time.
    """

    try:
        import adios2
    except ImportError as error:
        raise RuntimeError("the ADIOS2 Python bindings are required") from error

    adios = adios2.Adios(comm=comm)
    io = adios.declare_io(f"athenak_{Path(filename).stem}_{id(arrays)}")
    variables = {}
    definition_shapes = {}
    for name, input_array in arrays.items():
        array = np.ascontiguousarray(input_array)
        # An empty rank still participates in collective Open/Close.  Give its
        # unused definition a nonzero first dimension because some ADIOS2
        # versions reject a local-array Count containing zero.
        definition = array
        if not array.shape[0]:
            definition = np.empty((1, *array.shape[1:]), dtype=array.dtype)
        variables[name] = io.define_variable(
            name, definition, [], [], list(definition.shape), False
        )
        definition_shapes[name] = definition.shape
    for name, value in (attributes or {}).items():
        io.define_attribute(name, value)

    engine = io.open(str(filename), adios2.Mode.Write)
    local_totals = {name: 0 for name in arrays}
    try:
        engine.begin_step()
        blocks = (arrays,) if array_blocks is None else array_blocks
        for block in blocks:
            active = bool(
                block["points"].shape[0] and block["connectivity"].shape[0]
            )
            if not active:
                continue
            if set(block) != set(variables):
                raise ValueError("all ADIOS2 blocks must contain the same variables")
            for name, variable in variables.items():
                array = np.ascontiguousarray(block[name])
                if array.shape != definition_shapes[name]:
                    raise ValueError(
                        f"ADIOS2 local blocks for {name} have inconsistent shapes: "
                        f"{array.shape} != {definition_shapes[name]}"
                    )
                engine.put(variable, array, adios2.Mode.Sync)
                local_totals[name] += int(array.shape[0])
        engine.end_step()
    finally:
        engine.close()
    return {
        name: comm.allreduce(local_total)
        for name, local_total in local_totals.items()
    }


def _field(name: str, association: str, is_vector: bool) -> dict:
    return {
        "name": name.split("/", 1)[-1],
        "association": association,
        "array": {
            "array_type": "basic",
            "data_source": "source",
            "variable": name,
            "is_vector": "true" if is_vector else "false",
        },
    }


def fides_model(bp_filename: str | Path, arrays: Mapping[str, np.ndarray], cell_type: str) -> dict:
    """Build a Fides JSON data model for an explicit single-type cell set."""

    fields = []
    for name, array in arrays.items():
        if name.startswith("point_data/"):
            fields.append(_field(name, "points", array.ndim == 2))
        elif name.startswith("cell_data/"):
            fields.append(_field(name, "cell_set", array.ndim == 2))
    return {
        "AthenaK": {
            "data_sources": [{
                "name": "source",
                "filename_mode": "relative",
                "filename": Path(bp_filename).name,
            }],
            "coordinate_system": {"array": {
                "array_type": "basic",
                "data_source": "source",
                "variable": "points",
                "is_vector": "true",
                "static": True,
            }},
            "cell_set": {
                "cell_set_type": "single_type",
                "cell_type": cell_type,
                "data_source": "source",
                "variable": "connectivity",
                "static": True,
            },
            "fields": fields,
        }
    }


def write_fides_json(filename: str | Path, model: dict) -> Path:
    """Create (without replacing) an external Fides data-model file."""

    path = Path(filename)
    with path.open("x") as handle:
        json.dump(model, handle, indent=2)
        handle.write("\n")
    return path
