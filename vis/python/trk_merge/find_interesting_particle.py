#!/usr/bin/env python3
"""Search through tracks and find interesting examples with fast changes in
magnetic moment while in regions of larger magnetic-field-line curvature."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import h5py
import numpy as np


# Athena history headers label columns like "# [6]=B^2".
HEADER_RE = re.compile(r"\[(\d+)\]=([^\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks_h5")
    parser.add_argument("--b-rms", type=float)
    parser.add_argument("--history-file", type=Path)
    parser.add_argument("--min-mass", type=float)
    parser.add_argument("--mass-log-spacing", type=float)
    parser.add_argument(
        "--smooth",
        type=int,
        default=256,
        help="window size to use for rolling mean",
    )
    parser.add_argument(
        "--kappa-threshold",
        type=float,
        default=300.0,
        help="consider only values with a scaled, smooth kappa above this value",
    )
    parser.add_argument(
        "--delta-mu_m-threshold",
        type=float,
        default=1.0,
        help="consider only drops of log(mu_m) above this value",
    )
    return parser.parse_args()


def h5_string(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def particle_mass(species: int, run_dir: Path, opt: argparse.Namespace) -> float:
    min_mass = opt.min_mass
    spacing = opt.mass_log_spacing

    # The merged tracks store particle species, not particle mass. Production
    # runs write the mass ladder into a small runtime_overrides file.
    if min_mass is None or spacing is None:
        for path in sorted(run_dir.glob("*.runtime_overrides.txt")):
            for line in path.read_text().splitlines():
                if line.lstrip().startswith("#"):
                    continue
                key, separator, value = line.partition("=")
                if not separator:
                    continue
                if min_mass is None and key.strip() == "particles/min_mass":
                    min_mass = float(value)
                if spacing is None and key.strip() == "particles/mass_log_spacing":
                    spacing = float(value)
            if min_mass is not None and spacing is not None:
                break

    if min_mass is None or spacing is None:
        raise SystemExit(
            "could not infer particle mass ladder; pass --min-mass and "
            "--mass-log-spacing"
        )
    return min_mass * spacing**species


def history_columns(path: Path) -> dict[str, int]:
    columns = {}
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        for number, name in HEADER_RE.findall(line):
            # Header columns are 1-based; numpy arrays are 0-based.
            columns[name] = int(number) - 1
    return columns


def b_rms_from_history(path: Path) -> tuple[float, Path, float]:
    columns = history_columns(path)
    data = np.loadtxt(path, comments="#")
    data = np.atleast_2d(data)
    row = data[np.argmin(np.abs(data[:, 0]))]
    if "B^2" in columns:
        return float(np.sqrt(row[columns["B^2"]])), path, float(row[0])
    if {"1-ME", "2-ME", "3-ME"} <= set(columns):
        magnetic_energy = (
            row[columns["1-ME"]] + row[columns["2-ME"]] + row[columns["3-ME"]]
        )
        return float(np.sqrt(2.0 * magnetic_energy)), path, float(row[0])
    raise SystemExit(f"could not infer B_rms from {path}")


def get_b_rms(
    opt: argparse.Namespace
) -> tuple[float, Path | None, float | None]:
    if opt.b_rms is not None:
        return opt.b_rms, None, None
    if opt.history_file is not None:
        return b_rms_from_history(opt.history_file)
    print(
        "WARNING: no --history-file or --b-rms supplied; using B_rms=1.0. "
        "The curvature normalization is not physically scaled.",
        file=sys.stderr,
        flush=True,
    )
    return 1.0, None, None


def rolling_mean(y: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return y
    width = min(width, y.size)
    left = width // 2
    right = width - 1 - left
    kernel = np.ones(width) / width
    return np.convolve(np.pad(y, (left, right), mode="edge"), kernel, mode="valid")

def read_data(
    handle: h5py.File,
    opt: argparse.Namespace,
    time: np.array,
    brms: float,
    row: int,
):
    fields = h5_string(handle["values"].attrs["fields"]).split(",")
    field = {name: i for i, name in enumerate(fields)}

    data = handle["values"][row, :, :]
    particle = handle["particles"][row]

    source_run_dir = handle.attrs.get("source_run_dir", Path(opt.tracks_h5).parent)
    run_dir = Path(h5_string(source_run_dir))

    velocity = data[:, [field["vx"], field["vy"], field["vz"]]]
    magnetic_field = data[:, [field["bx"], field["by"], field["bz"]]]
    curvature = data[:, [field["k1"], field["k2"], field["k3"]]]

    bmag = np.linalg.norm(magnetic_field, axis=1)
    v2 = np.sum(velocity * velocity, axis=1)
    vpar = np.sum(velocity * magnetic_field, axis=1) / np.maximum(bmag, 1.0e-30)
    mu_m = np.maximum(v2 - vpar * vpar, 1.0e-30) / np.maximum(2.0 * bmag, 1.0e-30)

    mass = particle_mass(int(particle["species"]), run_dir, opt)
    # In these units, 2 pi c / Omega = 2 pi m / B_rms.
    gyro_period_length = 2.0 * np.pi * mass / brms
    kappa = np.linalg.norm(curvature, axis=1)
    kappa_scaled = np.maximum(kappa * gyro_period_length, 1.0e-30)

    mu_m_smooth = rolling_mean(mu_m, opt.smooth)
    kappa_smooth = rolling_mean(kappa, opt.smooth)
    kappa_scaled_smooth = rolling_mean(kappa_scaled, opt.smooth)

    order = np.argsort(kappa_smooth)
    for candidate in order[::-1]:
        k = kappa_smooth[candidate]
        if k < opt.kappa_threshold:
            return
        mu_m_1 = mu_m_smooth[candidate]
        mu_m_2 = mu_m_smooth[np.min([candidate + 2*opt.smooth, mu_m_smooth.size - 1])]
        delta_mu = np.abs(np.log10(mu_m_1) - np.log10(mu_m_2))
        if delta_mu > opt.delta_mu_m_threshold:
            print(f'{row},{time[candidate]},{k},{kappa_scaled_smooth[candidate]},{delta_mu}')
            return


def main() -> None:
    opt = parse_args()

    with h5py.File(opt.tracks_h5, "r") as handle:
        time = handle["times"][:]

        num_particles = handle["values"].shape[0]

        brms, history_path, history_time = get_b_rms(opt)

        print("row,time,kappa,kappa_scaled,delta_log_mu")
        for row in range(num_particles):
            read_data(handle, opt, time, brms, row)


if __name__ == "__main__":
    main()
