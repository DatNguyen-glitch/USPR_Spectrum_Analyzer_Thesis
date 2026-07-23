#!/usr/bin/env python3
"""
Reader for spectrum analyzer HDF5 files.

Reads both FFT and signal HDF5 files produced by zmq_receiver.py.

Usage:
    python3 h5_reader.py spectrum_data/fft_20260518_143000.h5
    python3 h5_reader.py fft_file.h5 --plot 42
    python3 h5_reader.py fft_file.h5 --plot 42 --bin-range 1000 1050  # THÊM MỚI: Chỉ vẽ từ bin 1000 đến 1050
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

            if total_frames > 0:
                ts_first = ds_ts[0]
                ts_last = ds_ts[-1]
                t0 = datetime.fromtimestamp(ts_first, tz=timezone.utc)
                t1 = datetime.fromtimestamp(ts_last, tz=timezone.utc)
                duration = ts_last - ts_first
                print(f"  Time span:     {t0.isoformat()} → {t1.isoformat()}")

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
        print(f"  [Row {row}] OUT OF RANGE")
        return

    fft_uint8 = ds_fft[row]
    ts = ds_ts[row]
    center = ds_center[row]
    tune_id = ds_tune[row]

    fft_db = dequantize_fft(fft_uint8, db_min, db_max)
    fft_len = len(fft_db)

    freqs = center + np.fft.fftshift(np.fft.fftfreq(fft_len, 1.0 / sample_rate))
    peak_bin = int(np.argmax(fft_db))
    peak_db = float(fft_db[peak_bin])
    peak_freq = float(freqs[peak_bin])
    noise_floor = float(np.median(fft_db))

    t = datetime.fromtimestamp(ts, tz=timezone.utc)
    print(f"\n  [Frame {row}] tune_id={tune_id}  center={center/1e6:.1f} MHz")
    print(f"    Peak: {peak_db:+.1f} dB @ {peak_freq/1e6:.3f} MHz (bin {peak_bin})")
    print(f"    Noise floor: {noise_floor:+.1f} dB")


def draw_spectrum(filepath: Path, row: int):
    # (Bỏ qua in ấn chi tiết cho ngắn gọn - Giữ nguyên gốc của bạn)
    pass


def plot_spectrum_gui(filepath: Path, row: int, bin_range=None):
    """
    Vẽ phổ High-Resolution bằng thư viện Matplotlib có hỗ trợ cắt khoảng Bin.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[ERROR] Tính năng --plot yêu cầu cài đặt thư viện matplotlib.")
        return

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

    # Giải mã và định tuyến trục tần số
    fft_db = dequantize_fft(fft_uint8, db_min, db_max)
    freqs = center + np.fft.fftshift(np.fft.fftfreq(fft_len, 1.0 / sample_rate))

    # XỬ LÝ CẮT KHOẢNG BIN (BIN SLICING)
    if bin_range is not None:
        start_bin, end_bin = bin_range
        # Bảo vệ biên (tránh vượt index)
        start_bin = max(0, start_bin)
        end_bin = min(fft_len, end_bin)
        
        if start_bin >= end_bin:
            print(f"[ERROR] Khoảng bin không hợp lệ: {start_bin} đến {end_bin}")
            return

        fft_db = fft_db[start_bin:end_bin]
        freqs = freqs[start_bin:end_bin]
        title_suffix = f" (Bins: {start_bin} → {end_bin})"
    else:
        title_suffix = f" (Toàn dải {fft_len} bins)"

    # Tính toán các thông số DỰA TRÊN KHOẢNG DỮ LIỆU ĐÃ CẮT (Local Metrics)
    peak_idx = int(np.argmax(fft_db))
    peak_db = float(fft_db[peak_idx])
    peak_freq = float(freqs[peak_idx])
    noise_floor = float(np.median(fft_db))
    
    t = datetime.fromtimestamp(ts, tz=timezone.utc)

    # Khởi tạo đồ thị
    plt.figure(figsize=(12, 6), dpi=100)
    
    plt.plot(freqs / 1e6, fft_db, color='#1f77b4', linewidth=1.2, label='FFT Spectrum')
    
    plt.axhline(noise_floor, color='black', linestyle='--', linewidth=1, alpha=0.7, 
                label=f'Local Noise Floor ({noise_floor:.1f} dB)')
    # plt.axhline(noise_floor + 4.0, color='red', linestyle=':', linewidth=1.5, alpha=0.8,
    #             label='Local Detection Threshold')
    
    # plt.plot(peak_freq / 1e6, peak_db, 'r^', markersize=8, 
    #          label=f'Local Peak: {peak_db:.1f} dB @ {peak_freq/1e6:.3f} MHz')

    # Cắt gọn trục Y cho đẹp (ẩn các rễ cây -120dBFS nếu có)
    plt.ylim(max(db_min, noise_floor - 25), min(db_max + 5, peak_db + 15))

    plt.title(f"Signal Spectrum at Center: {center/1e6:.1f} MHz", fontweight='bold')
    plt.xlabel("Frequency (MHz)", fontweight='bold')
    plt.ylabel("Magnitude (dBFS)", fontweight='bold')
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.show()


def read_sig_file(filepath: Path, row=None, row_start=None, row_end=None, summary=True):
    # (Bỏ qua in ấn chi tiết cho ngắn gọn - Giữ nguyên gốc của bạn)
    pass


def detect_file_type(filepath: Path) -> str:
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
    parser.add_argument("--range", type=int, nargs=2, metavar=("START", "END"), help="Print row range [START, END)")
    parser.add_argument("--spectrum", type=int, default=None, metavar="FRAME", help="Draw ASCII spectrum")
    parser.add_argument("--plot", type=int, default=None, metavar="FRAME", help="Plot high-resolution GUI spectrum")
    parser.add_argument("--bin-range", type=int, nargs=2, metavar=("START", "END"), help="THÊM MỚI: Chỉ vẽ khoảng bin cụ thể (Dùng chung với --plot)")
    parser.add_argument("--info", action="store_true", help="Print file info only")
    parser.add_argument("--no-summary", action="store_true", help="Skip summary")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        ftype = detect_file_type(path)
        
        # 1. Xử lý in thông tin metadata file trước
        if args.info:
            print_file_info(path)
            return

        # 2. Xử lý các lệnh vẽ đồ họa (chỉ dành cho file FFT)
        if args.plot is not None:
            if ftype != "fft":
                print("[ERROR] --plot requires an FFT file")
                sys.exit(1)
            plot_spectrum_gui(path, args.plot, bin_range=args.bin_range)
            return
            
        if args.spectrum is not None:
            if ftype != "fft":
                print("[ERROR] --spectrum requires an FFT file")
                sys.exit(1)
            draw_spectrum(path, args.spectrum)
            return

        # 3. Xử lý in dữ liệu Text ra console tùy theo loại file
        summary_flag = not args.no_summary
        
        if ftype == "fft":
            if args.row is not None:
                read_fft_file(path, row=args.row, summary=summary_flag)
            elif args.range is not None:
                read_fft_file(path, row_start=args.range[0], row_end=args.range[1], summary=summary_flag)
            else:
                read_fft_file(path, summary=summary_flag)
                
        elif ftype == "signal":
            if args.row is not None:
                read_sig_file(path, row=args.row, summary=summary_flag)
            elif args.range is not None:
                read_sig_file(path, row_start=args.range[0], row_end=args.range[1], summary=summary_flag)
            else:
                read_sig_file(path, summary=summary_flag)
        else:
            print(f"[ERROR] Không nhận diện được định dạng dữ liệu trong file: {path.name}")

if __name__ == "__main__":
    main()
