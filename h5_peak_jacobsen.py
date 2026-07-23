#!/usr/bin/env python3
"""Estimate center frequency from FFT HDF5 of a CW sine using Parabolic Interpolation.

Parabolic interpolation fits a quadratic to 3 bins around the FFT magnitude peak
(k-1, k, k+1) directly on the dBFS (log-power) spectrum:

    delta = 0.5 * (P[k-1] - P[k+1]) / (P[k-1] - 2*P[k] + P[k+1])
    f_est = center_hz + (peak_bin + delta - N/2) * Fs/N

where P[k] are dBFS values. Consistent with the thesis reference formula.

Metrics: Bias, Sigma, MAE, RMSE — in both kHz and ppm.
Acceptance criterion: |Bias| < ±5 ppm  (AD9361 spec: ±25 ppm; system target: ±5 ppm).

Ref: E. Jacobsen & P. Kootsookos, "Fast, Accurate Frequency Estimators," IEEE SP Mag., 2007.
"""

import glob
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt


FREQS_MHZ  = [358.5, 366, 373.5, 381, 388.5, 396, 403.5, 411,
              418.5, 426, 433.5, 441, 448.5, 456, 463.5, 471, 478.5, 486, 493.5, 501]
DATA_DIR   = Path("/home/datnguyen/Desktop/USPR_Spectrum_Analyzer_Thesis/spectrum_data/Jun13")
RUN_SUFFIX = "RX25TX5_FreqAccu"
PPM_LIMIT  = 5.0    # Acceptance criterion: |bias_ppm| < PPM_LIMIT → PASS

# Accumulators for plot (populated in main)
_bias_khz  : list[float] = []
_sigma_khz : list[float] = []
_bias_ppm  : list[float] = []
_sigma_ppm : list[float] = []


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def dequantize_fft(fft_uint8: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    """Convert uint8 quantized FFT back to dBFS values."""
    return fft_uint8.astype(np.float32) / 255.0 * (db_max - db_min) + db_min


def parabolic_estimator(frame_db: np.ndarray, peak_bin: int, fft_len: int,
                        center_hz: float, sample_rate: float) -> float:
    """Sub-bin frequency estimator using parabolic interpolation on dBFS.

    Fits a quadratic to P[k-1], P[k], P[k+1] in log-power (dBFS) domain:

        delta = 0.5 * (P[k-1] - P[k+1]) / (P[k-1] - 2*P[k] + P[k+1])
        f_est = center_hz + (peak_bin + delta - N/2) * Fs/N

    Applying the quadratic fit to dBFS (log scale) is the standard parabolic
    interpolation form and is consistent with the thesis reference formula.
    delta is clamped to [-0.5, 0.5] to guard against noise-corrupted neighbours.

    Args:
        frame_db   : full FFT frame in dBFS, fftshifted (length = fft_len)
        peak_bin   : argmax bin index inside frame_db
        fft_len    : FFT length N
        center_hz  : hardware centre frequency for this frame (Hz)
        sample_rate: ADC sample rate Fs (Hz)

    Returns:
        Estimated signal frequency in Hz.
    """
    if peak_bin <= 0 or peak_bin >= fft_len - 1:
        # Edge guard: fall back to raw bin centre frequency
        return center_hz + (peak_bin - fft_len / 2.0) * (sample_rate / fft_len)

    P_km1 = float(frame_db[peak_bin - 1])   # P[k-1]
    P_k   = float(frame_db[peak_bin])        # P[k]   ← peak
    P_kp1 = float(frame_db[peak_bin + 1])   # P[k+1]

    denominator = P_km1 - 2.0 * P_k + P_kp1
    if abs(denominator) < 1e-9:
        # Flat neighbourhood (degenerate case) — no sub-bin correction
        delta = 0.0
    else:
        delta = 0.5 * (P_km1 - P_kp1) / denominator

    delta = float(np.clip(delta, -0.5, 0.5))   # valid range: ±0.5 bin

    bin_resolution = sample_rate / fft_len
    return center_hz + (peak_bin + delta - fft_len / 2.0) * bin_resolution


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_all_frames(filepath: Path, true_freq_hz: float) -> dict:
    """Process every frame, estimate frequency via parabolic interpolation,
    reject 3-sigma outliers, and return accuracy metrics in Hz and ppm.

    Returns dict with keys:
        total_frames, rejected_frames, valid_frames,
        bias_hz, sigma_hz, mae_hz, rmse_hz,
        bias_ppm, sigma_ppm
    """
    with h5py.File(filepath, "r") as f:
        if "fft_data" not in f:
            raise ValueError("Not an FFT HDF5 file: missing 'fft_data'")

        ds_fft    = f["fft_data"]
        ds_center = f.get("center_hz", None)

        total_frames = int(ds_fft.shape[0])
        fft_len      = int(ds_fft.shape[1])
        db_min       = float(f.attrs.get("db_min",      -120.0))
        db_max       = float(f.attrs.get("db_max",         0.0))
        sample_rate  = float(f.attrs.get("sample_rate",   50e6))

        delta_f_list: list[float] = []
        half_window = 30    # ±30 bins ≈ ±730 kHz search window around target

        for row in range(total_frames):
            fft_row   = ds_fft[row]
            center_hz = float(ds_center[row]) if ds_center is not None else 0.0

            # Build fftshifted frequency axis and find target bin
            freq_axis  = center_hz + np.fft.fftshift(
                             np.fft.fftfreq(fft_len, 1.0 / sample_rate))
            target_bin = int(np.argmin(np.abs(freq_axis - true_freq_hz)))

            # Search window: min=1, max=fft_len-1 (slice excludes hi → max index = fft_len-2)
            lo = max(1,           target_bin - half_window)
            hi = min(fft_len - 1, target_bin + half_window)
            if lo >= hi:
                continue

            # Decode uint8 → dBFS
            frame_db = (dequantize_fft(fft_row, db_min, db_max)
                        if fft_row.dtype == np.uint8
                        else np.asarray(fft_row, dtype=np.float32))

            # Peak bin (absolute index in full frame)
            peak_bin = lo + int(np.argmax(frame_db[lo:hi]))
            if peak_bin <= 0 or peak_bin >= fft_len - 1:
                continue

            est_hz = parabolic_estimator(
                frame_db, peak_bin, fft_len, center_hz, sample_rate)
            delta_f_list.append(est_hz - true_freq_hz)

    # ---- Statistics -------------------------------------------------------
    _nan = float("nan")
    if not delta_f_list:
        return {
            "total_frames": total_frames, "rejected_frames": 0, "valid_frames": 0,
            "bias_hz": _nan, "sigma_hz": _nan, "mae_hz": _nan, "rmse_hz": _nan,
            "bias_ppm": _nan, "sigma_ppm": _nan,
        }

    arr = np.asarray(delta_f_list, dtype=np.float64)

    # 3-sigma outlier rejection
    raw_mean = np.mean(arr)
    raw_std  = np.std(arr)
    keep     = (arr >= raw_mean - 3 * raw_std) & (arr <= raw_mean + 3 * raw_std)
    clean    = arr[keep] if keep.any() else arr
    rejected = int(np.sum(~keep))

    bias_hz  = float(np.mean(clean))
    sigma_hz = float(np.std(clean))
    mae_hz   = float(np.mean(np.abs(clean)))
    rmse_hz  = float(np.sqrt(np.mean(clean ** 2)))

    # ppm = (error_hz / true_freq_hz) × 1e6
    bias_ppm  = bias_hz  / true_freq_hz * 1e6
    sigma_ppm = sigma_hz / true_freq_hz * 1e6

    return {
        "total_frames":    total_frames,
        "rejected_frames": rejected,           # 3σ-rejected frames, NOT physical frame loss
        "valid_frames":    len(clean),
        "bias_hz":   bias_hz,   "sigma_hz":  sigma_hz,
        "mae_hz":    mae_hz,    "rmse_hz":   rmse_hz,
        "bias_ppm":  bias_ppm,  "sigma_ppm": sigma_ppm,
    }


# ---------------------------------------------------------------------------
# File lookup
# ---------------------------------------------------------------------------

def find_fft_file_for_freq(f_mhz: float) -> Optional[Path]:
    files = sorted(glob.glob(str(DATA_DIR / f"{f_mhz}Mhz_{RUN_SUFFIX}" / "fft_*")))
    return Path(files[0]) if files else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    results      : list[tuple[float, dict]] = []
    total_reject = 0
    total_all    = 0

    # ---- Table header ----
    # COL = "{:>7}  {:>16}  {:>11}  {:>10}  {:>10}  {:>10}  {:>5}"
    # SEP = "-" * 76
    # print(COL.format("Freq", "3σ-rejected", "Bias", "Sigma",
    #                  "Bias", "Sigma", "Pass?"))
    # print(COL.format("(MHz)", "(frames)", "(kHz)", "(kHz)",
    #                  "(ppm)", "(ppm)", ""))
    COL = "{:>7}  {:>11}  {:>10}  {:>10}  {:>10}"
    SEP = "-" * 76
    print(COL.format("Freq", "Bias", "Sigma",
                     "Bias", "Sigma"))
    print(COL.format("(MHz)", "(kHz)", "(kHz)",
                     "(ppm)", "(ppm)"))
    print(SEP)

    for f_mhz in FREQS_MHZ:
        filepath = find_fft_file_for_freq(f_mhz)
        if filepath is None:
            print(f"[WARN] No file for {f_mhz} MHz")
            continue

        try:
            s = process_all_frames(filepath, f_mhz * 1e6)
        except Exception as e:
            print(f"[ERROR] {f_mhz} MHz — {e}")
            continue

        results.append((f_mhz, s))
        total_reject += s["rejected_frames"]
        total_all    += s["total_frames"]

        rej_str   = f"{s['rejected_frames']}/{s['total_frames']}"
        pass_flag = "PASS" if abs(s["bias_ppm"]) < PPM_LIMIT else "FAIL"

        _bias_khz.append(s["bias_hz"]  / 1e3)
        _sigma_khz.append(s["sigma_hz"] / 1e3)
        _bias_ppm.append(s["bias_ppm"])
        _sigma_ppm.append(s["sigma_ppm"])

        print(COL.format(
            f"{f_mhz:.1f}",
            # rej_str,
            f"{s['bias_hz']/1e3:+.3f}",
            f"{s['sigma_hz']/1e3:.3f}",
            f"{s['bias_ppm']:+.3f}",
            f"{s['sigma_ppm']:.3f}",
            pass_flag,
        ))

    if not results:
        print("No FFT files processed.")
        return 1

    # ---- Summary ----
    mean_mae_hz   = float(np.nanmean([s["mae_hz"]   for _, s in results]))
    mean_rmse_hz  = float(np.nanmean([s["rmse_hz"]  for _, s in results]))
    mean_bias_ppm = float(np.nanmean([s["bias_ppm"] for _, s in results]))
    mean_sigma_ppm = float(np.nanmean([s["sigma_ppm"] for _, s in results]))
    pass_count    = sum(1 for _, s in results if abs(s["bias_ppm"]) < PPM_LIMIT)

    print(SEP)
    print("SYSTEM SUMMARY")
    print(f"  Frequencies tested : {len(results)}")
    rej_pct = total_reject / total_all * 100 if total_all else 0
    # print(f"  3σ-rejected        : {total_reject}/{total_all} ({rej_pct:.2f}%)")
    print(f"  Mean MAE           : {mean_mae_hz/1e3:.3f} kHz")
    print(f"  Mean RMSE          : {mean_rmse_hz/1e3:.3f} kHz")
    print(f"  Mean Bias          : {mean_bias_ppm:+.3f} ppm")
    print(f"  Mean Sigma         : {mean_sigma_ppm:+.3f} ppm")
    # print(f"  Acceptance ±{PPM_LIMIT:.0f} ppm  : {pass_count}/{len(results)} PASS")

# ---- Plot: Cấu trúc 2 Subplots xếp chồng (Share X-axis) ----
    freqs = [f for f, _ in results]

    # Tạo Figure với 2 đồ thị con xếp chồng dọc, chung trục X
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=300)

    # --- SUBPLOT 1: SAI SỐ TUYỆT ĐỐI (kHz) ---
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
    ax1.errorbar(freqs, _bias_khz, yerr=_sigma_khz,
                 fmt='-o', color='#1f77b4', ecolor='#d62728',
                 elinewidth=1, capsize=3, markersize=4,
                 label=r'Bias $\pm 1\sigma$ (kHz)')
    ax1.set_ylabel('Absolute Error (kHz)', fontsize=5, fontweight='bold', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=5)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=5)
    
    ax1.set_title('Absolute Frequency Accuracy', fontsize=5, alpha=0.7)

    # --- SUBPLOT 2: SAI SỐ TƯƠNG ĐỐI (ppm) VÀ TIÊU CHUẨN NGHIỆM THU ---
    ax2.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
    ax2.errorbar(freqs, _bias_ppm, yerr=_sigma_ppm,
                 fmt='-s', color='#ff7f0e', ecolor='#9467bd',
                 elinewidth=1, capsize=3, markersize=4,
                 label=r'Bias $\pm 1\sigma$ (ppm)')
    
    # Vẽ đường biên giới hạn nghiệm thu công nghiệp
    ax2.axhline(PPM_LIMIT, color='red', linestyle=':', linewidth=1.2, alpha=0.8)
    ax2.axhline(-PPM_LIMIT, color='red', linestyle=':', linewidth=1.2, alpha=0.8,
                label=f'Acceptance Criterion (±{PPM_LIMIT:.0f} ppm)')
    
    ax2.set_xlabel('Carrier Frequency (MHz)', fontsize=5, fontweight='bold')
    ax2.set_ylabel('Relative Error (ppm)', fontsize=5, fontweight='bold', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e', labelsize=5)
    ax2.tick_params(axis='x', labelsize=5)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='lower right', fontsize=5)
    
    ax2.set_title('Normalized Frequency Accuracy', fontsize=5, alpha=0.7)

    # Tối ưu khoảng cách giữa các đồ thị con
    plt.suptitle('System Frequency Accuracy Evaluation\n on dequantized dBFS Data',
                 fontsize=9, fontweight='bold', y=0.97)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8, hspace=0.35, left=0.08)
    
    plt.savefig('Freq_Accuracy_Subplots_Presentation.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())