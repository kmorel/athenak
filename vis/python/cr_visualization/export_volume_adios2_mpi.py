#!/usr/bin/env python3
"""Export a distributed AthenaK MHD volume as ADIOS2/Fides."""

from __future__ import annotations

import argparse
from itertools import chain
from pathlib import Path
import sys


PYTHON_VIS = Path(__file__).resolve().parents[1]
if str(PYTHON_VIS) not in sys.path:
    sys.path.insert(0, str(PYTHON_VIS))

from cr_visualization.adios2_export import (  # noqa: E402
    fides_rectilinear_model,
    mesh_schema,
    output_paths,
    prepare_mesh_arrays,
    write_bp,
    write_fides_json,
)
from cr_visualization.cr_data import (  # noqa: E402
    discover_rank_files,
    read_rank_meshblocks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhd-rank0", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="output prefix")
    parser.add_argument("--quantities", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import adios2  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "export_volume_adios2_mpi.py requires the ADIOS2 Python bindings"
        ) from error
    try:
        from mpi4py import MPI
    except ImportError as error:
        raise SystemExit("export_volume_adios2_mpi.py requires mpi4py") from error

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    paths = output_paths(args.output)
    mesh_bp = paths["mesh_bp"].resolve()
    mesh_json = paths["mesh_json"].resolve()
    exists = mesh_bp.exists() or mesh_json.exists() if rank == 0 else False
    if comm.bcast(exists, root=0):
        raise SystemExit(f"refusing to replace an existing export for {args.output}")
    if rank == 0:
        mesh_bp.parent.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    rank_files = discover_rank_files(args.mhd_rank0) if rank == 0 else None
    rank_files = [Path(path) for path in comm.bcast(rank_files, root=0)]
    meshblocks = []
    for filename in rank_files[rank::comm.Get_size()]:
        meshblocks.extend(read_rank_meshblocks(filename, quantities=args.quantities))

    local_schema = mesh_schema(meshblocks[0]) if meshblocks else None
    schemas = comm.allgather(local_schema)
    schema = next((item for item in schemas if item is not None), None)
    if schema is None:
        raise SystemExit("the input contains no MeshBlocks")

    mesh_arrays = prepare_mesh_arrays(meshblocks[:1], schema)
    local_blocks = len(meshblocks)
    local_cells = sum(
        block[block["VariableNames"][0]].size for block in meshblocks
    )
    remaining_arrays = (
        prepare_mesh_arrays([block], schema) for block in meshblocks[1:]
    )
    write_bp(
        mesh_bp,
        mesh_arrays,
        comm,
        {"format": "athenak_fides_rectilinear_mesh_v2"},
        array_blocks=chain((mesh_arrays,), remaining_arrays) if meshblocks else (),
    )
    total_blocks = comm.allreduce(local_blocks)
    total_cells = comm.allreduce(local_cells)
    if rank == 0:
        write_fides_json(
            mesh_json, fides_rectilinear_model(mesh_bp, mesh_arrays)
        )
        print(
            f"{args.output}: MeshBlocks={total_blocks} mesh_cells={total_cells}"
        )


if __name__ == "__main__":
    main()
