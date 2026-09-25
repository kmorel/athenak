#!/usr/bin/env python3
"""Export distributed particle tracks as ADIOS2/Fides."""

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
    output_paths,
    prepare_track_arrays,
    visualization_track_fields,
    write_bp,
    write_fides_json,
)
from cr_visualization.cr_data import read_merged_track_partition  # noqa: E402

def intlist(s: str) -> list:
    return [int(num) for num in s.split()]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-tracks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="output prefix")
    parser.add_argument("--track-fields", nargs="+")
    parser.add_argument("--time-min", type=float)
    parser.add_argument("--time-max", type=float)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--particle-batch", type=int, default=4)
    parser.add_argument(
        "--source-rows", 
        type=intlist,
        help="Space separated list of tracks to export identified by row number",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        help="maximum number of tracks to export, sampled across the input",
    )
    parser.add_argument(
        "--domain-bounds",
        nargs=6,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="domain bounds used to detect jumps across periodic boundaries",
    )
    parser.add_argument(
        "--periodic-axes",
        nargs="+",
        choices=("x", "y", "z"),
        default=(),
        help="periodic axes whose boundary-crossing segments should be omitted",
    )
    args = parser.parse_args()
    if args.periodic_axes and args.domain_bounds is None:
        parser.error("--periodic-axes requires --domain-bounds")
    if args.max_tracks is not None and args.max_tracks < 1:
        parser.error("--max-tracks must be at least one")
    return args


def main() -> None:
    args = parse_args()
    try:
        import adios2  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "export_tracks_adios2_mpi.py requires the ADIOS2 Python bindings"
        ) from error
    try:
        from mpi4py import MPI
    except ImportError as error:
        raise SystemExit("export_tracks_adios2_mpi.py requires mpi4py") from error

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    paths = output_paths(args.output)
    tracks_bp = paths["tracks_bp"].resolve()
    tracks_json = paths["tracks_json"].resolve()
    exists = tracks_bp.exists() or tracks_json.exists() if rank == 0 else False
    if comm.bcast(exists, root=0):
        raise SystemExit(f"refusing to replace an existing export for {args.output}")
    if rank == 0:
        tracks_bp.parent.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    fields = visualization_track_fields(args.track_fields)
    tracks = read_merged_track_partition(
        args.merged_tracks,
        partition=rank,
        num_partitions=comm.Get_size(),
        time_min=args.time_min,
        time_max=args.time_max,
        time_stride=args.time_stride,
        fields=fields,
        particle_batch=args.particle_batch,
        source_rows=args.source_rows,
        max_tracks=args.max_tracks,
    )
    domain_bounds = (
        np.asarray(args.domain_bounds).reshape(3, 2)
        if args.domain_bounds is not None
        else None
    )
    axis_indices = tuple("xyz".index(axis) for axis in args.periodic_axes)
    track_arrays = prepare_track_arrays(tracks, domain_bounds, axis_indices)
    totals = write_bp(
        tracks_bp,
        track_arrays,
        comm,
        {"format": "athenak_fides_tracks_v1"},
    )
    if rank == 0:
        write_fides_json(
            tracks_json, fides_model(tracks_bp, track_arrays, "line")
        )
        print(
            f"{args.output}: track_points={totals['points']} "
            f"track_segments={totals['connectivity'] // 2}"
        )


if __name__ == "__main__":
    main()
