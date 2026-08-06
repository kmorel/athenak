#!/usr/bin/env python3
"""Export a distributed AthenaK snapshot and tracks as ADIOS2/Fides."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PYTHON_VIS = Path(__file__).resolve().parents[1]
if str(PYTHON_VIS) not in sys.path:
    sys.path.insert(0, str(PYTHON_VIS))

from cr_visualization.adios2_export import (  # noqa: E402
    fides_model,
    mesh_schema,
    output_paths,
    prepare_mesh_arrays,
    prepare_track_arrays,
    visualization_track_fields,
    write_bp,
    write_fides_json,
)
from cr_visualization.read_full_dataset_mpi import read_local_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhd-rank0", required=True, type=Path)
    parser.add_argument("--merged-tracks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="output prefix")
    parser.add_argument("--quantities", nargs="+")
    parser.add_argument("--track-fields", nargs="+")
    parser.add_argument("--time-min", type=float)
    parser.add_argument("--time-max", type=float)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--particle-batch", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import adios2  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "export_full_dataset_adios2_mpi.py requires the ADIOS2 Python bindings"
        ) from error
    try:
        from mpi4py import MPI
    except ImportError as error:
        raise SystemExit(
            "export_full_dataset_adios2_mpi.py requires mpi4py"
        ) from error

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    paths = {name: path.resolve() for name, path in output_paths(args.output).items()}
    exists = any(path.exists() for path in paths.values()) if rank == 0 else False
    if comm.bcast(exists, root=0):
        raise SystemExit(f"refusing to replace an existing export for {args.output}")
    if rank == 0:
        paths["mesh_bp"].parent.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    args.track_fields = visualization_track_fields(args.track_fields)
    meshblocks, tracks = read_local_data(args, comm)

    local_mesh_schema = mesh_schema(meshblocks[0]) if meshblocks else None
    schemas = comm.allgather(local_mesh_schema)
    resolved_mesh_schema = next((schema for schema in schemas if schema is not None), None)
    if resolved_mesh_schema is None:
        raise SystemExit("the input contains no MeshBlocks")
    geometry = None
    if meshblocks:
        geometry = (
            np.asarray(meshblocks[0].get("DomainBounds", meshblocks[0]["Bounds"])),
            tuple(meshblocks[0].get("PeriodicAxes", ())),
        )
    geometries = comm.allgather(geometry)
    domain_bounds, periodic_axes = next(item for item in geometries if item is not None)

    mesh_arrays = prepare_mesh_arrays(meshblocks[:1], resolved_mesh_schema)
    track_arrays = prepare_track_arrays(tracks, domain_bounds, periodic_axes)

    mesh_totals = write_bp(
        paths["mesh_bp"], mesh_arrays, comm,
        {"format": "athenak_fides_mesh_v1"},
        array_blocks=(
            prepare_mesh_arrays([block], resolved_mesh_schema)
            for block in meshblocks
        ),
    )
    track_totals = write_bp(
        paths["tracks_bp"], track_arrays, comm,
        {"format": "athenak_fides_tracks_v1"},
    )
    if rank == 0:
        write_fides_json(
            paths["mesh_json"], fides_model(paths["mesh_bp"], mesh_arrays, "hexahedron")
        )
        write_fides_json(
            paths["tracks_json"], fides_model(paths["tracks_bp"], track_arrays, "line")
        )
        print(
            f"{args.output}: mesh_points={mesh_totals['points']} "
            f"mesh_cells={mesh_totals['connectivity'] // 8} "
            f"track_points={track_totals['points']} "
            f"track_segments={track_totals['connectivity'] // 2}"
        )


if __name__ == "__main__":
    main()
