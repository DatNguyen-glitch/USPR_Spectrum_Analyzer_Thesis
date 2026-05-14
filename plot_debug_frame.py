#!/usr/bin/env python3
"""Plot a saved debug FFT frame produced by usrp_sweep_gr_rx.py.

Usage:
  python3 plot_debug_frame.py
  python3 plot_debug_frame.py debug_frames/tune_0001_center_500.0MHz_frame_05.npz
  python3 plot_debug_frame.py debug_frames/*.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEBUG_DIR = ROOT / "debug_frames"
# Dán path file .npz vào đây nếu muốn mở trực tiếp, ví dụ:
# INPUT_PATH = "/home/datnguyen/Desktop/USPR_Spectrum_Analyzer_Thesis/debug_frames/tune_0001_center_500.0MHz_frame_05.npz"
INPUT_PATH = "/home/datnguyen/Desktop/USPR_Spectrum_Analyzer_Thesis/debug_frames/tune_0206_center_410.0MHz_frame_05.npz"


def resolve_input_paths(argv: list[str]) -> list[Path]:
    if INPUT_PATH.strip():
        return [Path(INPUT_PATH).expanduser().resolve()]

    if len(argv) > 1:
        paths = [Path(arg).expanduser().resolve() for arg in argv[1:]]
    else:
        paths = sorted(DEBUG_DIR.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz files found in {DEBUG_DIR}")
        paths = [paths[-1]]

    expanded: list[Path] = []
    for path in paths:
        if any(ch in str(path) for ch in "*?[]"):
            expanded.extend(sorted(Path().glob(str(path))))
        else:
            expanded.append(path)

    if not expanded:
        raise FileNotFoundError("No matching debug frame files found")

    return expanded


def load_frame(path: Path):
    data = np.load(path, allow_pickle=False)
    frame = data["frame"].astype(np.complex64, copy=False)
    center_hz = float(data["center_hz"])
    sample_rate = float(data["sample_rate"])
    tune_id = int(data["tune_id"])
    frame_epoch = int(data["frame_epoch"])
    return frame, center_hz, sample_rate, tune_id, frame_epoch


def plot_frame(ax, path: Path):
    frame, center_hz, sample_rate, tune_id, frame_epoch = load_frame(path)

    fft_len = frame.size
    freq_axis_hz = center_hz + np.fft.fftshift(np.fft.fftfreq(fft_len, d=1.0 / sample_rate))
    mag_db = 20.0 * np.log10(np.abs(np.fft.fftshift(frame)) + 1e-12)

    ax.plot(freq_axis_hz / 1e6, mag_db, linewidth=1.0)
    ax.set_title(
        f"{path.name} | tune_id={tune_id} | frame_epoch={frame_epoch} | center={center_hz/1e6:.1f} MHz"
    )
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3)


def main() -> int:
    paths = resolve_input_paths(sys.argv)
    fig, axes = plt.subplots(len(paths), 1, figsize=(14, 4 * len(paths)), squeeze=False)

    for ax, path in zip(axes[:, 0], paths):
        plot_frame(ax, path)

    fig.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())