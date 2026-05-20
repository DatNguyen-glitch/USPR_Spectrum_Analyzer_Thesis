#!/usr/bin/env python3
"""
Reader for spectrum analyzer HDF5 files.

Reads both FFT and signal HDF5 files produced by zmq_receiver.py.

Usage:
    python3 h5_reader.py spectrum_data/fft_20260518_143000.h5
    python3 h5_reader.py spectrum_data/signals_20260518_143000.h5
    python3 h5_reader.py spectrum_data/              # read all files in directory
    python3 h5_reader.py fft_file.h5 --row 100      # print single FFT frame
    python3 h5_reader.py fft_file.h5 --range 0 50   # print frames 0-49
    python3 h5_reader.py fft_file.h5 --spectrum 42   # detailed ASCII spectrum of frame 42
    python3 h5_reader.py sig_file.h5 --row 5        # print single signal event
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def dequantize_fft(fft_uint8: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    """Convert uint8 quantized FFT back to dB values."""
    return fft_uint8.astype(np.float32) / 255.0 * (db_max - db_min) + db_min


def print_file_info(filepath: Path):
    """Print file-level metadata."""
    with h5py.File(filepath, "r") as f:
        print(f"File: {filepath.name}")
        print(f"  Size: {filepath.stat().st_size / 1024:.1f} KB")
        print(f"  Datasets: {list(f.keys())}")
        print(f"  Attributes:")
        for k, v in f.attrs.items():
            print(f"    {k} = {v}")
        for name, ds in f.items():
            print(f"  /{name}: shape={ds.shape} dtype={ds.dtype}")


def read_fft_file(filepath: Path, row=None, row_start=None, row_end=None, summary=True):
    """Read and display FFT HDF5 file contents."""
    with h5py.File(filepath, "r") as f:
        ds_fft = f["fft_data"]
        ds_ts = f["timestamps"]
        ds_center = f["center_hz"]
        ds_tune = f["tune_id"]

        total_frames = ds_fft.shape[0]
        fft_len = ds_fft.shape[1]
        db_min = f.attrs.get("db_min", -120.0)
        db_max = f.attrs.get("db_max", 0.0)
        sample_rate = f.attrs.get("sample_rate", 50e6)

        if summary:
            print(f"\n{'='*60}")
            print(f"FFT File: {filepath.name}")
            print(f"{'='*60}")
            print(f"  Total frames:  {total_frames}")
            print(f"  FFT length:    {fft_len}")
            print(f"  Sample rate:   {sample_rate/1e6:.1f} MHz")
            print(f"  dB range:      [{db_min}, {db_max}]")
            print(f"  Quantization:  uint8 (error ±{(db_max-db_min)/255:.2f} dB)")

            if total_frames > 0:
                ts_first = ds_ts[0]
                ts_last = ds_ts[-1]
                t0 = datetime.fromtimestamp(ts_first, tz=timezone.utc)
                t1 = datetime.fromtimestamp(ts_last, tz=timezone.utc)
                duration = ts_last - ts_first
                print(f"  Time span:     {t0.isoformat()} → {t1.isoformat()}")
                print(f"  Duration:      {duration:.1f} s")
                if duration > 0:
                    print(f"  Avg rate:      {total_frames/duration:.1f} fps")

                # Frequency coverage
                centers = ds_center[:]
                unique_centers = np.unique(centers)
                print(f"  Center freqs:  {len(unique_centers)} unique")
                print(f"  Freq range:    {unique_centers.min()/1e6:.1f} - {unique_centers.max()/1e6:.1f} MHz")

        # Print specific row(s)
        if row is not None:
            _print_fft_frame(f, row, db_min, db_max, sample_rate)
        elif row_start is not None:
            end = min(row_end or row_start + 10, total_frames)
            for r in range(row_start, end):
                _print_fft_frame(f, r, db_min, db_max, sample_rate)

        return total_frames


def _print_fft_frame(f, row, db_min, db_max, sample_rate):
    """Print a single FFT frame."""
    ds_fft = f["fft_data"]
    ds_ts = f["timestamps"]
    ds_center = f["center_hz"]
    ds_tune = f["tune_id"]

    if row >= ds_fft.shape[0]:
        print(f"  [Row {row}] OUT OF RANGE (max {ds_fft.shape[0]-1})")
        return

    fft_uint8 = ds_fft[row]
    ts = ds_ts[row]
    center = ds_center[row]
    tune_id = ds_tune[row]

    fft_db = dequantize_fft(fft_uint8, db_min, db_max)
    fft_len = len(fft_db)

    # Compute frequency axis
    freqs = center + np.fft.fftshift(np.fft.fftfreq(fft_len, 1.0 / sample_rate))

    peak_bin = int(np.argmax(fft_db))
    peak_db = float(fft_db[peak_bin])
    peak_freq = float(freqs[peak_bin])
    noise_floor = float(np.median(fft_db))

    t = datetime.fromtimestamp(ts, tz=timezone.utc)
    print(f"\n  [Frame {row}] tune_id={tune_id}  center={center/1e6:.1f} MHz  {t.strftime('%H:%M:%S.%f')[:-3]}")
    print(f"    Peak: {peak_db:+.1f} dB @ {peak_freq/1e6:.3f} MHz (bin {peak_bin})")
    print(f"    Noise floor: {noise_floor:+.1f} dB")
    print(f"    Span: {freqs[0]/1e6:.1f} - {freqs[-1]/1e6:.1f} MHz")

    # ASCII spectrum (72 columns)
    cols = 72
    decimated = np.array([fft_db[i * fft_len // cols:(i + 1) * fft_len // cols].max()
                          for i in range(cols)])
    bar_min, bar_max = db_min, max(db_max, peak_db + 5)
    normalized = np.clip((decimated - bar_min) / (bar_max - bar_min), 0, 1)
    bar_chars = " ▁▂▃▄▅▆▇█"
    bar = "".join(bar_chars[int(v * (len(bar_chars) - 1))] for v in normalized)
    print(f"    [{bar}]")


def draw_spectrum(filepath: Path, row: int):
    """Draw a detailed ASCII spectrum plot for a single FFT frame."""
    with h5py.File(filepath, "r") as f:
        ds_fft = f["fft_data"]
        ds_ts = f["timestamps"]
        ds_center = f["center_hz"]
        ds_tune = f["tune_id"]

        total = ds_fft.shape[0]
        if row < 0 or row >= total:
            print(f"Frame {row} out of range (0-{total-1})")
            return

        fft_len = ds_fft.shape[1]
        db_min = f.attrs.get("db_min", -120.0)
        db_max = f.attrs.get("db_max", 0.0)
        sample_rate = f.attrs.get("sample_rate", 50e6)

        fft_uint8 = ds_fft[row]
        ts = ds_ts[row]
        center = ds_center[row]
        tune_id = ds_tune[row]

    fft_db = dequantize_fft(fft_uint8, db_min, db_max)
    freqs = center + np.fft.fftshift(np.fft.fftfreq(fft_len, 1.0 / sample_rate))

    peak_bin = int(np.argmax(fft_db))
    peak_db = float(fft_db[peak_bin])
    peak_freq = float(freqs[peak_bin])
    noise_floor = float(np.median(fft_db))
    t = datetime.fromtimestamp(ts, tz=timezone.utc)

    # Terminal dimensions
    try:
        term_cols = os.get_terminal_size().columns
    except OSError:
        term_cols = 120
    term_cols = max(80, term_cols)

    # Layout: left margin for dB labels + plot area
    label_w = 10   # "-120.0 |"
    plot_w = term_cols - label_w - 2  # 2 for right border

    # Decimate FFT to plot width (take max per bin group)
    bins_per_col = fft_len / plot_w
    plot_db = np.zeros(plot_w)
    plot_freq = np.zeros(plot_w)
    for i in range(plot_w):
        lo = int(i * bins_per_col)
        hi = int((i + 1) * bins_per_col)
        hi = max(hi, lo + 1)
        plot_db[i] = np.max(fft_db[lo:hi])
        mid = (lo + hi) // 2
        plot_freq[i] = freqs[min(mid, fft_len - 1)]

    # Y-axis range: snap to 10 dB grid
    y_top = float(np.ceil((peak_db + 5) / 10) * 10)
    y_bot = float(np.floor((noise_floor - 15) / 10) * 10)
    y_top = min(y_top, db_max + 10)
    y_bot = max(y_bot, db_min)
    if y_top <= y_bot:
        y_top = y_bot + 60

    # Plot height
    plot_h = 24

    # Header
    freq_lo = freqs[0] / 1e6
    freq_hi = freqs[-1] / 1e6
    print()
    print(f"  Frame {row}  |  tune_id={tune_id}  center={center/1e6:.1f} MHz  "
          f"{t.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC")
    print(f"  Peak: {peak_db:+.1f} dB @ {peak_freq/1e6:.3f} MHz  |  "
          f"Noise: {noise_floor:+.1f} dB  |  SNR: {peak_db - noise_floor:.1f} dB")
    print()

    # Draw plot rows (top to bottom)
    for r in range(plot_h):
        db_hi = y_top - r * (y_top - y_bot) / plot_h
        db_lo = y_top - (r + 1) * (y_top - y_bot) / plot_h

        # Y-axis label (show every 4th row)
        if r % 4 == 0:
            label = f"{db_hi:+7.1f} "
        else:
            label = "        "

        # Build row characters
        row_chars = []
        for c in range(plot_w):
            val = plot_db[c]
            if val >= db_hi:
                # Signal fills this cell fully
                row_chars.append("█")
            elif val >= db_lo:
                # Partial fill — pick sub-character
                frac = (val - db_lo) / (db_hi - db_lo)
                sub = " ▁▂▃▄▅▆▇█"
                row_chars.append(sub[int(frac * (len(sub) - 1))])
            else:
                row_chars.append(" ")

        # Mark threshold and noise floor lines
        line_char = None
        if db_lo <= noise_floor < db_hi:
            line_char = ("─", "N")
        elif db_lo <= (noise_floor + 4) < db_hi:  # threshold (noise + margin)
            line_char = ("┄", "T")

        if line_char:
            for c in range(plot_w):
                if row_chars[c] == " ":
                    row_chars[c] = line_char[0]

        line = "".join(row_chars)

        # Add noise/threshold marker at right edge
        marker = ""
        if line_char:
            marker = f" {line_char[1]}"

        print(f"{label}│{line}│{marker}")

    # X-axis border
    print(f"        └{'─' * plot_w}┘")

    # X-axis frequency labels
    n_labels = min(7, plot_w // 12)
    label_line = [" "] * plot_w
    for i in range(n_labels):
        col = int(i * (plot_w - 1) / max(1, n_labels - 1))
        freq_mhz = plot_freq[col] / 1e6
        txt = f"{freq_mhz:.1f}"
        start = max(0, col - len(txt) // 2)
        end = min(plot_w, start + len(txt))
        for j, ch in enumerate(txt[:end - start]):
            label_line[start + j] = ch
    print(f"         {''.join(label_line)}")
    print(f"         {'Frequency (MHz)':^{plot_w}}")

    # Legend
    print()
    print(f"  N = Noise floor ({noise_floor:+.1f} dB)    T = Threshold ({noise_floor + 4:+.1f} dB)")
    print(f"  Y range: [{y_bot:+.1f}, {y_top:+.1f}] dB    X range: [{freq_lo:.1f}, {freq_hi:.1f}] MHz")
    print(f"  Resolution: {sample_rate/fft_len/1e3:.1f} kHz/bin    Plot: {plot_w}x{plot_h} chars")


def read_sig_file(filepath: Path, row=None, row_start=None, row_end=None, summary=True):
    """Read and display signal detection HDF5 file contents."""
    with h5py.File(filepath, "r") as f:
        ds_events = f["events"]
        ds_ts = f["timestamps"]
        ds_center = f["center_hz"]

        total_events = ds_events.shape[0]

        if summary:
            print(f"\n{'='*60}")
            print(f"Signal File: {filepath.name}")
            print(f"{'='*60}")
            print(f"  Total events:  {total_events}")
            print(f"  Attributes:")
            for k, v in f.attrs.items():
                print(f"    {k} = {v}")

            if total_events > 0:
                ts_first = ds_ts[0]
                ts_last = ds_ts[-1]
                t0 = datetime.fromtimestamp(ts_first, tz=timezone.utc)
                t1 = datetime.fromtimestamp(ts_last, tz=timezone.utc)
                duration = ts_last - ts_first
                print(f"  Time span:     {t0.isoformat()} → {t1.isoformat()}")
                print(f"  Duration:      {duration:.1f} s")
                if duration > 0:
                    print(f"  Avg rate:      {total_events/duration:.1f} events/s")

                # Signal statistics
                centers = ds_center[:]
                unique_centers = np.unique(centers)
                print(f"  Center freqs:  {len(unique_centers)} unique")

                # Sample first few to count total signals
                sample_n = min(100, total_events)
                total_signals = 0
                for i in range(sample_n):
                    evt = json.loads(ds_events[i])
                    total_signals += len(evt.get("signals", []))
                avg_per_event = total_signals / sample_n
                print(f"  Avg signals/event: {avg_per_event:.1f} (sampled {sample_n})")

        # Print specific row(s)
        if row is not None:
            _print_sig_event(f, row)
        elif row_start is not None:
            end = min(row_end or row_start + 10, total_events)
            for r in range(row_start, end):
                _print_sig_event(f, r)

        return total_events


def _print_sig_event(f, row):
    """Print a single signal detection event."""
    ds_events = f["events"]
    ds_ts = f["timestamps"]

    if row >= ds_events.shape[0]:
        print(f"  [Event {row}] OUT OF RANGE (max {ds_events.shape[0]-1})")
        return

    raw = ds_events[row]
    ts = ds_ts[row]
    evt = json.loads(raw)

    t = datetime.fromtimestamp(ts, tz=timezone.utc)
    center = evt.get("center_hz", 0)
    noise = evt.get("noise_floor_db", 0)
    threshold = evt.get("threshold_db", 0)
    signals = evt.get("signals", [])

    print(f"\n  [Event {row}] {t.strftime('%H:%M:%S.%f')[:-3]}  center={center/1e6:.1f} MHz")
    print(f"    Noise floor: {noise:+.1f} dB  Threshold: {threshold:+.1f} dB")
    print(f"    Detected {len(signals)} signal(s):")

    for i, sig in enumerate(signals):
        print(
            f"      [{i}] freq={sig['peak_freq']/1e6:.3f} MHz  "
            f"power={sig['peak_db']:+.1f} dB  "
            f"SNR={sig['peak_snr_db']:.1f} dB  "
            f"BW={sig['width_hz']/1e3:.1f} kHz  "
            f"span=[{sig.get('span_start',0)/1e6:.3f}, {sig.get('span_end',0)/1e6:.3f}] MHz"
        )


def detect_file_type(filepath: Path) -> str:
    """Detect whether file is FFT or signal based on datasets."""
    with h5py.File(filepath, "r") as f:
        if "fft_data" in f:
            return "fft"
        elif "events" in f:
            return "signal"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Read spectrum analyzer HDF5 files")
    parser.add_argument("path", help="HDF5 file or directory containing HDF5 files")
    parser.add_argument("--row", type=int, default=None, help="Print specific row")
    parser.add_argument("--range", type=int, nargs=2, metavar=("START", "END"),
                        help="Print row range [START, END)")
    parser.add_argument("--spectrum", type=int, default=None, metavar="FRAME",
                        help="Draw detailed ASCII spectrum for a specific FFT frame")
    parser.add_argument("--info", action="store_true", help="Print file info only (no data)")
    parser.add_argument("--no-summary", action="store_true", help="Skip summary, show rows only")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_dir():
        files = sorted(path.glob("*.h5"))
        if not files:
            print(f"No .h5 files found in {path}")
            sys.exit(1)
        print(f"Found {len(files)} HDF5 files in {path}")
        for fp in files:
            ftype = detect_file_type(fp)
            if args.info:
                print_file_info(fp)
                print()
            elif args.spectrum is not None and ftype == "fft":
                draw_spectrum(fp, args.spectrum)
            elif ftype == "fft":
                read_fft_file(fp, row=args.row,
                              row_start=args.range[0] if args.range else None,
                              row_end=args.range[1] if args.range else None)
            elif ftype == "signal":
                read_sig_file(fp, row=args.row,
                              row_start=args.range[0] if args.range else None,
                              row_end=args.range[1] if args.range else None)
            else:
                print(f"  Unknown file type: {fp.name}")
    elif path.is_file():
        if args.info:
            print_file_info(path)
            sys.exit(0)

        ftype = detect_file_type(path)
        summary = not args.no_summary

        if args.spectrum is not None:
            if ftype != "fft":
                print("--spectrum requires an FFT file")
                sys.exit(1)
            draw_spectrum(path, args.spectrum)
        elif ftype == "fft":
            read_fft_file(path, row=args.row,
                          row_start=args.range[0] if args.range else None,
                          row_end=args.range[1] if args.range else None,
                          summary=summary)
        elif ftype == "signal":
            read_sig_file(path, row=args.row,
                          row_start=args.range[0] if args.range else None,
                          row_end=args.range[1] if args.range else None,
                          summary=summary)
        else:
            print(f"Unknown file type. Datasets found:")
            print_file_info(path)
            sys.exit(1)
    else:
        print(f"Path not found: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
