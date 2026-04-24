#!/usr/bin/env python3
"""Generate synthetic HD-EMG sleeve data and mapping artifacts.

Outputs (default: ../data):
- synthetic_signal.npy: float32 array of shape (frames, channels)
- ring_map.npy: int32 26x5 map with channel indices and -1 for invalid cells
- valid_mask.npy: bool 26x5 mask of valid map cells
- metadata.json: dataset metadata and generation settings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_default_ring_map() -> tuple[np.ndarray, np.ndarray]:
    """Build the exact 26x5 sleeve ring map used by the geometry-aware model."""
    strip_a = list(range(1, 14))
    strip_b = list(range(14, 27))
    strip_c = list(range(27, 40))
    strip_d = list(range(40, 53))
    strip_e = list(range(53, 65))
    strip_f = list(range(65, 77))
    strip_g = list(range(77, 90))
    strip_h = list(range(90, 103))
    strip_i = list(range(103, 116))
    strip_j = list(range(116, 129))

    rings_1_based = [
        [strip_a[0], strip_c[0], strip_g[0], strip_i[0], -1],
        [strip_b[0], strip_d[0], strip_h[0], strip_j[0], -1],
    ]

    for idx in range(1, 13):
        rings_1_based.append(
            [strip_a[idx], strip_c[idx], strip_e[idx - 1], strip_g[idx], strip_i[idx]]
        )
        rings_1_based.append(
            [strip_b[idx], strip_d[idx], strip_f[idx - 1], strip_h[idx], strip_j[idx]]
        )

    ring_map = np.asarray(rings_1_based, dtype=np.int32)
    valid_mask = ring_map > 0
    ring_map = ring_map - 1
    ring_map[~valid_mask] = -1
    return ring_map, valid_mask


def generate_signal(frames: int, channels: int, fs_hz: int, seed: int) -> np.ndarray:
    """Generate synthetic EMG-like signals: noise + bursts + oscillatory components."""
    rng = np.random.default_rng(seed)
    t = np.arange(frames, dtype=np.float32) / np.float32(fs_hz)

    signal = (0.02 * rng.standard_normal((frames, channels))).astype(np.float32)

    # Shared temporal bursts with channel-wise spatial scaling.
    bursts = [(0.35, 0.08, 0.20), (0.62, 0.06, 0.28), (0.82, 0.05, 0.22)]
    for center_s, width_s, gain in bursts:
        env = gain * np.exp(-0.5 * ((t - center_s) / width_s) ** 2)
        spatial = (0.6 + 0.8 * rng.random(channels)).astype(np.float32)
        signal += np.outer(env.astype(np.float32), spatial)

    # Channel-specific high-frequency activity.
    freqs = rng.uniform(40.0, 120.0, size=channels).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=channels).astype(np.float32)
    for ch in range(channels):
        signal[:, ch] += (0.015 * np.sin(2.0 * np.pi * freqs[ch] * t + phases[ch])).astype(np.float32)

    return signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic HD-EMG sleeve dataset")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: ../data)")
    parser.add_argument("--frames", type=int, default=900, help="Number of time frames")
    parser.add_argument("--channels", type=int, default=128, help="Number of channels")
    parser.add_argument("--fs-hz", type=int, default=1000, help="Sampling frequency in Hz")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir if args.output_dir is not None else (script_dir.parent / "data")
    output_dir.mkdir(parents=True, exist_ok=True)

    ring_map, valid_mask = build_default_ring_map()
    valid_count = int(valid_mask.sum())

    if valid_count != args.channels:
        raise ValueError(
            f"Valid-mask count mismatch: expected {args.channels}, found {valid_count}. "
            "Adjust mapping parameters or channel count."
        )

    signal = generate_signal(args.frames, args.channels, args.fs_hz, args.seed)

    metadata = {
        "dataset_name": "synthetic_hdemg_sleeve",
        "frames": int(args.frames),
        "channels": int(args.channels),
        "fs_hz": int(args.fs_hz),
        "seed": int(args.seed),
        "layout": "26x5 default sleeve ring map",
        "map_shape": [int(ring_map.shape[0]), int(ring_map.shape[1])],
        "valid_count": valid_count,
        "invalid_count": int((~valid_mask).sum()),
        "files": {
            "signal": "synthetic_signal.npy",
            "ring_map": "ring_map.npy",
            "valid_mask": "valid_mask.npy",
        },
    }

    np.save(output_dir / "synthetic_signal.npy", signal)
    np.save(output_dir / "ring_map.npy", ring_map)
    np.save(output_dir / "valid_mask.npy", valid_mask)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote dataset to: {output_dir}")
    print(f"signal shape: {signal.shape}")
    print(f"valid count: {valid_count}")
    print(f"frame count: {args.frames}")


if __name__ == "__main__":
    main()
