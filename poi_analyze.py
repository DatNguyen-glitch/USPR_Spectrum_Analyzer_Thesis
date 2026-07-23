#!/usr/bin/env python3
"""
POI analysis v2 — corrected implementation.

Chạy:
  python3 poi_analyze_v2_fixed.py --h5-dir ./poi_data \
      --burst-csv ./poi_data/poi_tx_b5ms_*.csv --f-burst 436e6

Tuỳ chọn:
  --tx-off-dir ./txoff_data
  --tol-ms 3
  --threshold-db 12
"""

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple
from scipy.stats import beta

import h5py
import numpy as np

# Ghi toàn bộ đầu ra màn hình vào file log cùng nội dung
try:
    _LOG_PATH = os.path.join(os.path.dirname(__file__), "poi_analyze.log")
    _log_file = open(_LOG_PATH, "a", buffering=1, encoding="utf-8")

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, data):
            for s in self._streams:
                try:
                    s.write(data)
                except Exception:
                    pass

        def flush(self):
            for s in self._streams:
                try:
                    s.flush()
                except Exception:
                    pass

        def isatty(self):
            for s in self._streams:
                try:
                    if s.isatty():
                        return True
                except Exception:
                    pass
            return False

        @property
        def encoding(self):
            return getattr(self._streams[0], "encoding", "utf-8")

    sys.stdout = _Tee(sys.stdout, _log_file)
    sys.stderr = _Tee(sys.stderr, _log_file)
except Exception as _e:
    # Nếu không thể mở file log, tiếp tục bình thường và in cảnh báo ra stderr gốc.
    try:
        sys.__stderr__.write(f"[WARN] Không thể mở poi_analyze.log: {_e}\n")
    except Exception:
        pass


# ── Hằng số phần cứng: khóa theo clock ADC, không suy ra từ timestamp ────────
SAMP_RATE = 50e6
FFT_LEN = 2048
FPGA_DECIM = 50
T_S = FFT_LEN * FPGA_DECIM / SAMP_RATE      # 2.048 ms
T_INT = FFT_LEN / SAMP_RATE                 # 40.96 us
DC_GUARD = 60
BURST_HALF = 3
FAR_MARGIN = 0.050                          # 50 ms

SWEEP_START_HZ = 358e6
SWEEP_STOP_HZ = 508e6
SWEEP_STEP_HZ = 25e6

SWEEP_HOPS_HZ = np.arange(
    SWEEP_START_HZ,
    SWEEP_STOP_HZ + 0.5 * SWEEP_STEP_HZ,
    SWEEP_STEP_HZ,
    dtype=np.float64,
)

def log(message: str = "") -> None:
    print(message, flush=True)


def hdr(title: str) -> None:
    log("\n" + "═" * 76)
    log(title)
    log("═" * 76)


def bin_of(f_target: float, f_center: float) -> int:
    return int(round(FFT_LEN / 2 + (f_target - f_center) / (SAMP_RATE / FFT_LEN)))


def load_bursts(path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    t_on: List[float] = []
    t_off: List[float] = []
    b_ms: List[float] = []

    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            t_on.append(float(row["t_on"]))
            t_off.append(float(row["t_off"]))
            b_ms.append(float(row["b_ms"]))

    if not t_on:
        sys.exit(f"[FATAL] burst CSV rỗng: {path}")

    on = np.asarray(t_on, dtype=np.float64)
    off = np.asarray(t_off, dtype=np.float64)
    order = np.argsort(on)
    on = on[order]
    off = off[order]

    if np.any(off <= on):
        sys.exit("[FATAL] có burst với t_off <= t_on")

    return on, off, float(np.median(np.asarray(b_ms, dtype=np.float64)))


def load_h5(directory: str) -> Dict[str, object]:
    files = sorted(glob.glob(os.path.join(directory, "fft_*.h5")))
    if not files:
        sys.exit(f"[FATAL] không có fft_*.h5 trong {directory}")

    fft_parts: List[np.ndarray] = []
    ts_parts: List[np.ndarray] = []
    center_parts: List[np.ndarray] = []
    tune_parts: List[np.ndarray] = []
    epoch_parts: List[np.ndarray] = []
    attrs: Dict[str, object] = {}
    has_epoch = True
    nonempty_files = 0

    for file_path in files:
        with h5py.File(file_path, "r") as h5:
            if not attrs:
                attrs = dict(h5.attrs)

            n_rows = int(h5["fft_data"].shape[0])
            if n_rows == 0:
                continue

            nonempty_files += 1
            fft_parts.append(h5["fft_data"][:])
            ts_parts.append(h5["timestamps"][:])
            center_parts.append(h5["center_hz"][:])
            tune_parts.append(h5["tune_id"][:])

            if "epoch" in h5:
                epoch_parts.append(h5["epoch"][:])
            else:
                has_epoch = False

    if not fft_parts:
        sys.exit(f"[FATAL] tất cả fft_*.h5 trong {directory} đều rỗng")

    # Giữ nguyên thứ tự ghi trong file. Không sort toàn bộ theo software timestamp,
    # vì timestamp có thể jitter hoặc gom lô và làm đảo thứ tự frame trong dwell.
    return {
        "fft": np.concatenate(fft_parts),
        "ts": np.concatenate(ts_parts),
        "center": np.concatenate(center_parts),
        "tune": np.concatenate(tune_parts),
        "epoch": np.concatenate(epoch_parts) if has_epoch and epoch_parts else None,
        "attrs": attrs,
        "n_files": nonempty_files,
    }


def dwell_groups(data: Dict[str, object]) -> List[np.ndarray]:
    """Nhóm các row liên tiếp thuộc cùng một dwell theo (tune_id, center_hz)."""
    tune = np.asarray(data["tune"])
    center = np.asarray(data["center"])
    n = len(tune)
    if n == 0:
        return []

    boundaries = np.flatnonzero(
        (tune[1:] != tune[:-1]) | (center[1:] != center[:-1])
    ) + 1
    return [g for g in np.split(np.arange(n), boundaries) if len(g)]


def snr_of(
    fft_u8: np.ndarray,
    burst_bin: int,
    db_min: float,
    db_max: float,
) -> np.ndarray:
    """SNR tại vùng burst so với median noise floor của cùng frame."""
    if fft_u8.ndim != 2 or fft_u8.shape[1] != FFT_LEN:
        raise ValueError(f"fft_data phải có shape (N, {FFT_LEN})")

    scale = (db_max - db_min) / 255.0
    lo = max(0, burst_bin - BURST_HALF)
    hi = min(FFT_LEN, burst_bin + BURST_HALF + 1)

    peak = fft_u8[:, lo:hi].max(axis=1).astype(np.float32)

    mask = np.ones(FFT_LEN, dtype=bool)
    center_bin = FFT_LEN // 2
    mask[center_bin - DC_GUARD:center_bin + DC_GUARD] = False
    mask[lo:hi] = False
    noise = np.median(fft_u8[:, mask], axis=1).astype(np.float32)

    return (peak - noise) * scale


def owner_of(
    rf_start: np.ndarray,
    t_on: np.ndarray,
    t_off: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Burst index mà frame [rf, rf+T_INT] overlap; -1 nếu không overlap."""
    right_edge = rf_start + T_INT + tolerance
    owner = np.searchsorted(t_on, right_edge, side="right") - 1

    valid = owner >= 0
    owner = np.clip(owner, 0, len(t_on) - 1)
    overlap = (
        valid
        & (rf_start - tolerance < t_off[owner])
        & (rf_start + T_INT + tolerance > t_on[owner])
    )
    return np.where(overlap, owner, -1)


def estimate_delay(
    snr_hop: np.ndarray,
    ts_hop: np.ndarray,
    t_on: np.ndarray,
    t_off: np.ndarray,
) -> Dict[str, object]:
    """Ước lượng software delay bằng score overlap của các frame SNR cao."""
    if len(snr_hop) == 0:
        raise ValueError("không có frame tại hop burst")

    k = min(len(snr_hop), max(100, int(0.01 * len(snr_hop))))
    candidate_idx = np.argpartition(snr_hop, -k)[-k:]
    candidates = ts_hop[candidate_idx]

    delays = np.arange(-0.050, 0.200, 0.00025)
    scores = np.asarray([
        np.count_nonzero(owner_of(candidates - delay, t_on, t_off, 0.0) >= 0)
        for delay in delays
    ])

    peak_idx = int(np.argmax(scores))
    delay = float(delays[peak_idx])
    peak_score = int(scores[peak_idx])

    outside = np.abs(delays - delay) > 0.002
    second_score = int(scores[outside].max()) if np.any(outside) else 0
    at_boundary = peak_idx in (0, len(delays) - 1)
    ambiguous = second_score >= 0.95 * peak_score if peak_score > 0 else True

    return {
        "delay": delay,
        "peak_score": peak_score,
        "second_score": second_score,
        "n_candidates": k,
        "at_boundary": at_boundary,
        "ambiguous": ambiguous,
    }

def estimate_delay_robust(
    snr_hop: np.ndarray,
    ts_hop: np.ndarray,
    t_on: np.ndarray,
    t_off: np.ndarray,
) -> Dict[str, object]:
    """
    Tìm delay bằng burst-level capped score.

    Với mỗi delay:
      1. Gán frame vào burst theo overlap thời gian.
      2. Mỗi burst chỉ giữ SNR lớn nhất.
      3. Clip SNR tại 15 dB để interferer 40–47 dB không chi phối.
      4. Score = tổng phần SNR vượt 6 dB.

    Không dùng top-1% frame.
    """

    n_bursts = len(t_on)
    association_tolerance = 0.25e-3
    score_floor_db = 6.0
    score_cap_db = 15.0

    def evaluate_grid(
        delays: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        scores = np.zeros(len(delays), dtype=np.float64)
        hit8 = np.zeros(len(delays), dtype=np.int64)
        hit10 = np.zeros(len(delays), dtype=np.int64)
        associated = np.zeros(len(delays), dtype=np.int64)

        for i, delay in enumerate(delays):
            rf_time = ts_hop - delay

            owners = owner_of(
                rf_time,
                t_on,
                t_off,
                association_tolerance,
            )

            valid = owners >= 0

            if not np.any(valid):
                continue

            # Mỗi burst chỉ giữ frame có SNR cao nhất.
            burst_max_snr = np.full(
                n_bursts,
                -np.inf,
                dtype=np.float64,
            )

            np.maximum.at(
                burst_max_snr,
                owners[valid],
                snr_hop[valid],
            )

            present = np.isfinite(burst_max_snr)
            values = burst_max_snr[present]

            associated[i] = int(np.count_nonzero(present))
            hit8[i] = int(np.count_nonzero(values > 8.0))
            hit10[i] = int(np.count_nonzero(values > 10.0))

            clipped = np.clip(
                values,
                score_floor_db,
                score_cap_db,
            )

            scores[i] = float(
                np.sum(clipped - score_floor_db)
            )

        return scores, hit8, hit10, associated

    # Coarse search: ±2 giây, bước 1 ms.
    coarse_delays = np.arange(
        -2.0,
        2.000001,
        1e-3,
        dtype=np.float64,
    )

    (
        coarse_scores,
        coarse_hit8,
        coarse_hit10,
        coarse_associated,
    ) = evaluate_grid(coarse_delays)

    coarse_best_index = int(np.argmax(coarse_scores))
    coarse_delay = float(coarse_delays[coarse_best_index])

    # Fine search: ±3 ms quanh coarse peak, bước 0.05 ms.
    fine_delays = coarse_delay + np.arange(
        -3e-3,
        3.0001e-3,
        0.05e-3,
        dtype=np.float64,
    )

    (
        fine_scores,
        fine_hit8,
        fine_hit10,
        fine_associated,
    ) = evaluate_grid(fine_delays)

    fine_best_index = int(np.argmax(fine_scores))
    best_delay = float(fine_delays[fine_best_index])
    best_score = float(fine_scores[fine_best_index])
    best_hit8 = int(fine_hit8[fine_best_index])
    best_hit10 = int(fine_hit10[fine_best_index])
    best_associated = int(fine_associated[fine_best_index])

    # Peak thứ hai phải nằm cách peak chính ít nhất 10 ms.
    outside_main_peak = np.abs(
        coarse_delays - best_delay
    ) > 10e-3

    if np.any(outside_main_peak):
        second_local_index = int(
            np.argmax(coarse_scores[outside_main_peak])
        )

        second_indices = np.flatnonzero(outside_main_peak)
        second_index = int(
            second_indices[second_local_index]
        )

        second_delay = float(
            coarse_delays[second_index]
        )
        second_score = float(
            coarse_scores[second_index]
        )
        second_hit8 = int(
            coarse_hit8[second_index]
        )
    else:
        second_delay = np.nan
        second_score = 0.0
        second_hit8 = 0

    # Robust significance so với toàn bộ các delay sai.
    score_median = float(np.median(coarse_scores))
    score_mad = float(
        np.median(
            np.abs(coarse_scores - score_median)
        )
    )

    if score_mad > 0:
        score_z = (
            best_score - score_median
        ) / (1.4826 * score_mad)
    else:
        score_z = np.inf if best_score > score_median else 0.0

    hit8_median = float(np.median(coarse_hit8))
    hit10_median = float(np.median(coarse_hit10))

    at_boundary = coarse_best_index in (
        0,
        len(coarse_delays) - 1,
    )

    peak_ratio = (
        best_score / second_score
        if second_score > 0
        else np.inf
    )

    ambiguous = (
        score_z < 5.0
        or peak_ratio < 1.10
        or best_hit8 < max(10, int(hit8_median + 5))
    )

    return {
        "delay": best_delay,

        # Giữ các key cũ để không phá step6().
        "peak_score": best_hit8,
        "second_score": second_hit8,
        "n_candidates": best_associated,
        "at_boundary": at_boundary,
        "ambiguous": ambiguous,

        # Diagnostic mới.
        "robust_score": best_score,
        "second_robust_score": second_score,
        "second_delay": second_delay,
        "score_median": score_median,
        "score_mad": score_mad,
        "score_z": score_z,
        "peak_ratio": peak_ratio,
        "hit8": best_hit8,
        "hit10": best_hit10,
        "hit8_median": hit8_median,
        "hit10_median": hit10_median,
        "n_associated": best_associated,
    }

def validate_phase_distribution(n_values: np.ndarray) -> Dict[str, object]:
    """Kiểm quan hệ count của cửa sổ dài L trên lưới chu kỳ T_S với pha đều."""
    if len(n_values) == 0:
        return {"ok": False, "reason": "không có dwell"}

    values, counts = np.unique(n_values, return_counts=True)
    probs = counts / counts.sum()
    expected = float(np.mean(n_values))
    n_floor = int(np.floor(expected))
    n_ceil = int(np.ceil(expected))
    frac = expected - n_floor

    if n_floor == n_ceil:
        ok = len(values) == 1 and int(values[0]) == n_floor
        return {
            "ok": ok,
            "values": values,
            "probs": probs,
            "expected": expected,
            "n_floor": n_floor,
            "n_ceil": n_ceil,
            "frac": 0.0,
            "p_ceil": 1.0 if ok else float(probs[values == n_ceil].sum()),
            "reason": "count nguyên cố định" if ok else "E[n] nguyên nhưng count không cố định",
        }

    allowed = {n_floor, n_ceil}
    support_ok = set(values.tolist()).issubset(allowed)
    p_ceil = float(probs[values == n_ceil].sum()) if n_ceil in values else 0.0
    probability_ok = abs(p_ceil - frac) < 0.05

    return {
        "ok": support_ok and probability_ok,
        "values": values,
        "probs": probs,
        "expected": expected,
        "n_floor": n_floor,
        "n_ceil": n_ceil,
        "frac": frac,
        "p_ceil": p_ceil,
        "reason": "support/count probability phù hợp" if support_ok and probability_ok
                  else "support hoặc xác suất count không phù hợp",
    }


# ═══ 1. SELF-CHECK ═══════════════════════════════════════════════════════════
def step1(D, t_on, t_off):
    hdr("1 · SELF-CHECK")

    a = D["attrs"]
    dbm = float(a.get("db_min", -120.0))
    dbM = float(a.get("db_max", 0.0))

    tx_start = float(t_on[0])
    tx_end = float(t_off[-1])
    rx_start = float(D["ts"].min())
    rx_end = float(D["ts"].max())
    edge_guard = 0.100

    covered = (
        (t_on >= rx_start + edge_guard) &
        (t_off <= rx_end - edge_guard)
    )

    n_original = len(t_on)
    t_on = t_on[covered]
    t_off = t_off[covered]

    log("\n── RX COVERAGE FILTER ──")
    log(f"  burst ban đầu        : {n_original}")
    log(f"  burst được RX bao phủ: {len(t_on)}")
    log(f"  burst bị loại        : {n_original - len(t_on)}")

    if len(t_on) == 0:
        log("[STOP] Không có burst nào nằm hoàn toàn trong RX coverage.")
        return

    log(f"  frames {len(D['ts']):,} từ {D['n_files']} file")
    log(f"  db_min/db_max : {dbm}/{dbM} "
        f"{'✔ từ attrs' if 'db_min' in a else '⚠ THIẾU attrs → mặc định'}")

    log("\n  ── TIMESTAMP RANGE CHECK ──")
    log(f"  TX first/last : {tx_start:.9f} → {tx_end:.9f}")
    log(f"  RX first/last : {rx_start:.9f} → {rx_end:.9f}")
    log(f"  TX duration   : {tx_end - tx_start:.3f} s")
    log(f"  RX duration   : {rx_end - rx_start:.3f} s")
    log(f"  RX−TX start   : {rx_start - tx_start:+.6f} s")
    log(f"  RX−TX end     : {rx_end - tx_end:+.6f} s")

    overlap_start = max(tx_start, rx_start)
    overlap_end = min(tx_end, rx_end)
    overlap = max(0.0, overlap_end - overlap_start)

    if overlap <= 0:
        log("  ✘ TX và RX KHÔNG có khoảng thời gian chồng lấn.")
        log("    Có thể sai file, khác clock domain, UTC/local-time hoặc relative/epoch time.")
        timeline_ok = False
    else:
        log(f"  timeline overlap: {overlap:.3f} s  ✔")
        timeline_ok = True

    if D["epoch"] is None:
        log("\n  epoch : ✘ KHÔNG CÓ → phải thu lại.")
        return dbm, dbM, False, timeline_ok

    log("\n  epoch : ✔ có")
    return dbm, dbM, True, timeline_ok


# ═══ 2. FRAME LOSS ═══════════════════════════════════════════════════════════
def step2(data: Dict[str, object]) -> Tuple[float, int]:
    hdr("2 · FRAME LOSS — quét epoch")
    epoch = np.asarray(data["epoch"])
    groups = dwell_groups(data)

    diff_parts: List[np.ndarray] = []
    for group in groups[:300]:
        sorted_epoch = np.sort(epoch[group])
        if len(sorted_epoch) > 1:
            diff_parts.append(np.diff(sorted_epoch))

    steps = np.concatenate(diff_parts) if diff_parts else np.asarray([1])
    positive_steps = steps[steps > 0]
    mode_step = int(np.bincount(positive_steps).argmax()) if len(positive_steps) else 1

    if mode_step == 1:
        log("  bước epoch = 1  ✔  (FFT_STREAM_DECIMATION = 1)")
    else:
        log(f"  ⚠ bước epoch phổ biến = {mode_step}")

    missing = 0
    duplicates = 0
    expected_total = 0

    for group in groups:
        unique_epoch = np.unique(epoch[group])
        duplicates += len(group) - len(unique_epoch)
        expected = int(unique_epoch.max()) + 1
        expected_total += expected
        missing += expected - len(unique_epoch)

    loss_rate = 100.0 * missing / expected_total if expected_total else 0.0
    log(f"  epoch trùng lặp : {duplicates}  {'✔' if duplicates == 0 else '⚠'}")
    log(f"  frame thiếu     : {missing:,}/{expected_total:,} = {loss_rate:.4f}%  "
        f"{'✔' if missing == 0 else '⚠'}")
    log("  (Cận dưới: frame cuối dwell bị mất không thể phát hiện chỉ bằng max(epoch).)")

    return loss_rate, duplicates


# ═══ 3. TIMING ═══════════════════════════════════════════════════════════════
def step3(
    data: Dict[str, object],
    f_burst: float,
    loss_rate: float,
    duplicates: int,
) -> Dict[str, object]:
    hdr("3 · TIMING")

    timestamps = np.asarray(data["ts"])
    center = np.asarray(data["center"])
    epoch = np.asarray(data["epoch"])
    raw_groups = dwell_groups(data)

    def is_sweep_center(value: float) -> bool:
        return bool(
            np.any(
                np.isclose(
                    value,
                    SWEEP_HOPS_HZ,
                    atol=1.0,
                    rtol=0.0,
                )
            )
        )

    # Bỏ các group warm-up, chẳng hạn center 500 MHz.
    all_groups = [
        group
        for group in raw_groups
        if is_sweep_center(float(center[group[0]]))
    ]

    observed_hops = np.unique(center)

    # Chỉ giữ các hop thuộc sweep thật và thực sự xuất hiện trong dữ liệu.
    hops = np.asarray(
        [
            hop
            for hop in SWEEP_HOPS_HZ
            if np.any(
                np.isclose(
                    observed_hops,
                    hop,
                    atol=1.0,
                    rtol=0.0,
                )
            )
        ],
        dtype=np.float64,
    )

    ignored_hops = [
        hop
        for hop in observed_hops
        if not is_sweep_center(float(hop))
    ]

    if ignored_hops:
        log(
            "  center ngoài sweep bị bỏ qua: "
            f"{[f'{hop / 1e6:.0f}' for hop in ignored_hops]} MHz"
        )

    hop_burst = float(hops[np.argmin(np.abs(hops - f_burst))])
    burst_groups = [g for g in all_groups if float(center[g[0]]) == hop_burst]

    if len(burst_groups) < 2:
        sys.exit("[FATAL] cần ít nhất hai lần ghé hop burst để đo T_sweep")

    log(f"  hop grid ({len(hops)}): {[f'{h / 1e6:.0f}' for h in hops]} MHz")
    log(
        f"  hop burst : {hop_burst / 1e6:.0f} MHz  "
        f"(lệch tâm {(f_burst - hop_burst) / 1e6:+.1f} MHz)"
    )

    dwell_epoch: List[np.ndarray] = []
    n_recorded: List[int] = []
    n_scheduled: List[int] = []

    for group in burst_groups:
        observed = np.sort(np.unique(epoch[group]))
        dwell_epoch.append(observed)
        n_recorded.append(len(observed))
        n_scheduled.append(int(observed.max()) + 1)

    n_rec = np.asarray(n_recorded, dtype=int)
    n_sched = np.asarray(n_scheduled, dtype=int)
    values, counts = np.unique(n_rec, return_counts=True)
    probs = counts / counts.sum()
    expected_recorded = float(np.mean(n_rec))

    log(
        f"\n  frame/dwell ghi được: "
        f"{dict(zip(values.tolist(), np.round(probs, 3).tolist()))}"
    )
    log(
        f"    E[n_recorded] = {expected_recorded:.3f} | "
        f"E[n_scheduled] = {np.mean(n_sched):.3f}"
    )

    # Mỗi dwell được đại diện bằng median timestamp. Không dùng gap dưới-dwell.
    visit_times = np.asarray([np.median(timestamps[group]) for group in burst_groups])
    visit_times.sort()
    sweep_diffs = np.diff(visit_times)
    sweep_diffs = sweep_diffs[sweep_diffs > 0]
    if len(sweep_diffs) == 0:
        sys.exit("[FATAL] không đo được T_sweep hợp lệ")

    t_sweep = float(np.median(sweep_diffs))
    t_hop = t_sweep / len(hops)

    # Diagnostic trung bình theo epoch. Prediction vẫn luôn dùng T_S lý thuyết.
    slopes: List[float] = []
    residuals: List[float] = []
    near_zero_gaps = 0
    total_gaps = 0

    for group in all_groups[:500]:
        e = epoch[group].astype(np.float64)
        u = timestamps[group].astype(np.float64)
        if len(e) <= 2 or e.max() <= e.min():
            continue

        first = int(np.argmin(e))
        last = int(np.argmax(e))
        slopes.append((u[last] - u[first]) / (e[last] - e[first]))

        intercept = float(np.median(u - e * T_S))
        residuals.extend((u - (intercept + e * T_S)).tolist())

        order = np.argsort(e)
        local_gaps = np.diff(u[order])
        near_zero_gaps += int(np.count_nonzero(np.abs(local_gaps) < 0.1 * T_S))
        total_gaps += len(local_gaps)

    t_s_measured = float(np.median(slopes)) if slopes else np.nan
    residual_array = np.asarray(residuals, dtype=np.float64)
    residual_p95 = (
        float(np.percentile(np.abs(residual_array), 95)) if len(residual_array) else np.nan
    )
    zero_gap_fraction = near_zero_gaps / total_gaps if total_gaps else np.nan

    log(f"\n  T_s lý thuyết : {T_S * 1e3:.4f} ms")
    log(f"  T_s slope      : {t_s_measured * 1e3:.4f} ms")
    log(f"  grid residual p95: {residual_p95 * 1e3:.3f} ms")
    log(f"  adjacent gap gần 0: {zero_gap_fraction * 100:.2f}%")
    log("  Prediction dùng T_s lý thuyết; các số trên chỉ diagnostic timestamp batching.")

    t_span = (expected_recorded - 1.0) * T_S + T_INT
    delta_span = len(hops) * t_span / t_sweep
    delta_integrate = len(hops) * expected_recorded * T_INT / t_sweep

    phase_result: Dict[str, object]
    if loss_rate > 0.0 or duplicates > 0:
        phase_result = {
            "ok": False,
            "reason": "không kiểm phase vì có frame loss hoặc duplicate",
        }
    else:
        phase_result = validate_phase_distribution(n_sched)

    if phase_result["ok"]:
        expected_scheduled = float(np.mean(n_sched))
        t_collect = expected_scheduled * T_S
        delta_collect = len(hops) * t_collect / t_sweep
    else:
        t_collect = np.nan
        delta_collect = np.nan

    log("\n  ── Các đại lượng timing ──")
    log(f"    T_frame_span  = {t_span * 1e3:.3f} ms")
    log(f"    δ_frame_span  = {delta_span:.5f}")
    log(f"    δ_integrate   = {delta_integrate:.5f}")

    if phase_result["ok"]:
        log(f"    phase check    : ✔ {phase_result['reason']}")
        log(f"    T_collect      = {t_collect * 1e3:.3f} ms")
        log(f"    δ_collect      = {delta_collect:.5f}")
    else:
        log(f"    phase check    : ⚠ {phase_result['reason']}")
        log("    T_collect/δ_collect không được sử dụng.")

    log(f"\n  T_hop = {t_hop * 1e3:.2f} ms | T_sweep = {t_sweep * 1e3:.2f} ms")

    return {
        "hops": hops,
        "hop_burst": hop_burst,
        "dwell_ep": dwell_epoch,
        "n_rec": n_rec,
        "n_sched": n_sched,
        "E_n": expected_recorded,
        "T_sweep": t_sweep,
        "T_hop": t_hop,
        "T_span": t_span,
        "d_span": delta_span,
        "T_collect": t_collect,
        "d_collect": delta_collect,
        "d_integrate": delta_integrate,
        "ok_phase": bool(phase_result["ok"]),
    }


# ═══ 4. ADJACENT-HOP RESPONSE ═══════════════════════════════════════════════
def step4(
    data: Dict[str, object],
    f_burst: float,
    timing: Dict[str, object],
    db_min: float,
    db_max: float,
    threshold: float,
    t_on: np.ndarray,
    t_off: np.ndarray,
    delay: float,
    tolerance: float,
) -> bool:
    hdr("4 · ADJACENT-HOP RESPONSE — burst-level check")

    center = np.asarray(data["center"])
    timestamps = np.asarray(data["ts"])
    fft = np.asarray(data["fft"])
    n_bursts = len(t_on)
    one_look_ok = True

    checked = 0
    for hop in np.asarray(timing["hops"]):
        offset_mhz = (f_burst - hop) / 1e6
        if abs(offset_mhz) > 25.5 or float(hop) == float(timing["hop_burst"]):
            continue

        burst_bin = bin_of(f_burst, float(hop))
        if not 0 <= burst_bin < FFT_LEN:
            continue

        selected = center == hop
        if not np.any(selected):
            continue

        checked += 1
        snr = snr_of(fft[selected], burst_bin, db_min, db_max)
        rf_time = timestamps[selected] - delay
        owners = owner_of(rf_time, t_on, t_off, tolerance)
        hot = snr > threshold

        adjacent_hit = np.zeros(n_bursts, dtype=bool)
        valid = (owners >= 0) & hot
        if np.any(valid):
            adjacent_hit[np.unique(owners[valid])] = True

        n_hit = int(adjacent_hit.sum())
        free_hot = int(np.count_nonzero(hot & (owners < 0)))
        rate = n_hit / n_bursts

        log(f"\n    hop {hop / 1e6:.0f} MHz | offset {offset_mhz:+.1f} MHz | bin {burst_bin}")
        log(f"      burst cũng được hop kề bắt : {n_hit}/{n_bursts} = {rate * 100:.2f}%")
        log(f"      hot frame không gắn burst  : {free_hot}")

        if rate >= 0.005:
            one_look_ok = False

    if checked == 0:
        log("\n  Không có hop kề nào đặt burst trong dải số ±25 MHz.")
    elif one_look_ok:
        log("\n  ✔ Adjacent-hop response không đáng kể; mô hình one-look/sweep hợp lệ.")
    else:
        log("\n  ⚠ Hop kề cũng bắt burst; mô hình one-look/sweep không mô tả toàn hệ thống.")
        log("    Main-hop-only POI bên dưới không bị double-count, nhưng không phải total-system POI.")

    return one_look_ok


# ═══ 5. THRESHOLD CALIBRATION ════════════════════════════════════════════════
def step5(
    snr_hop: np.ndarray,
    rf_hop: np.ndarray,
    t_on: np.ndarray,
    t_off: np.ndarray,
    n_bursts: int,
    tolerance: float,
    burst_duration: float,
    txoff_snr: Optional[np.ndarray] = None,
) -> Optional[float]:
    hdr("5 · HIỆU CHUẨN NGƯỠNG")

    if txoff_snr is not None:
        noise = np.asarray(txoff_snr)
        source = "run TX-off"
    else:
        owners = owner_of(rf_hop, t_on, t_off, FAR_MARGIN)
        noise = snr_hop[owners < 0]
        source = f"frame ngoài mọi burst với margin {FAR_MARGIN * 1e3:.0f} ms"

    if len(noise) == 0:
        log("  ✘ Không có mẫu noise để hiệu chuẩn threshold.")
        return None

    log(f"  noise samples: {len(noise):,} ({source})")
    log(
        f"  mean {noise.mean():+.2f} dB | std {noise.std():.2f} dB | "
        f"p99 {np.percentile(noise, 99):+.2f} dB"
    )

    candidate_frames = (burst_duration + T_INT + 2.0 * tolerance) / T_S
    budget = 0.005

    log(
        f"\n  cửa sổ association = "
        f"{(burst_duration + T_INT + 2 * tolerance) * 1e3:.3f} ms"
    )
    log(f"  candidate noise frames/burst ≈ {candidate_frames:.3f}")
    log(f"  false-positive budget ≤ {budget * 100:.2f}% burst")
    log(f"\n    {'threshold':>10} | {'FA/frame':>12} | {'FP bursts upper':>16}")
    log("    " + "-" * 46)

    selected_threshold: Optional[float] = None
    n_noise = len(noise)

    for threshold in range(4, 41, 2):
        exceedances = int(np.count_nonzero(noise > threshold))
        if exceedances >= n_noise:
            fa_upper = 1.0
        else:
            fa_upper = float(
                beta.ppf(
                    0.95,
                    exceedances + 1,
                    n_noise - exceedances,
                )
            )
        fp_burst_upper = 1.0 - (1.0 - fa_upper) ** candidate_frames
        expected_fp_bursts = n_bursts * fp_burst_upper

        mark = ""
        if selected_threshold is None and fp_burst_upper <= budget:
            selected_threshold = float(threshold)
            mark = "  ← CHỌN"

        log(
            f"    {threshold:8d} dB | {fa_upper * 100:10.5f}% | "
            f"{expected_fp_bursts:14.2f}{mark}"
        )

    if selected_threshold is None:
        log("\n  ✘ Không threshold nào đến 40 dB đạt false-positive budget.")
        return None

    log(f"\n  threshold được chọn: noise + {selected_threshold:.1f} dB")
    return selected_threshold


# ═══ 6. MEASURE POI ══════════════════════════════════════════════════════════
def step6(
    snr_hop: np.ndarray,
    ts_hop: np.ndarray,
    t_on: np.ndarray,
    t_off: np.ndarray,
    threshold: float,
    tolerance: float,
    delay_info: Dict[str, object],
) -> Dict[str, object]:
    hdr("6 · ĐO POI")

    delay = float(delay_info["delay"])
    log(
        f"  delay = {delay * 1e3:+.2f} ms | "
        f"best {delay_info['peak_score']}/{delay_info['n_candidates']} | "
        f"second {delay_info['second_score']}"
    )
    if delay_info["at_boundary"]:
        log("  ⚠ delay peak nằm ở biên search range")
    if delay_info["ambiguous"]:
        log("  ⚠ delay peak không duy nhất")

    rf_time = ts_hop - delay
    n_bursts = len(t_on)

    tolerance_grid = sorted(set([
        0.0,
        1e-3,
        2e-3,
        3e-3,
        5e-3,
        8e-3,
        12e-3,
        float(tolerance),
    ]))

    log(f"\n    {'tol':>8} | {'POI':>9} | {'delta':>10}")
    log("    " + "-" * 34)

    results: Dict[float, Tuple[float, int]] = {}
    previous: Optional[float] = None

    for current_tolerance in tolerance_grid:
        owners = owner_of(rf_time, t_on, t_off, current_tolerance)
        hit = np.zeros(n_bursts, dtype=bool)
        valid = (owners >= 0) & (snr_hop > threshold)
        if np.any(valid):
            hit[np.unique(owners[valid])] = True

        poi = float(hit.mean())
        results[current_tolerance] = (poi, int(hit.sum()))
        delta_text = "" if previous is None else f"{(poi - previous) * 100:+.2f} pt"
        log(f"    {current_tolerance * 1e3:6.1f} ms | {poi * 100:7.2f}% | {delta_text:>10}")
        previous = poi

    poi, n_hit = results[float(tolerance)]

    z = 1.96
    n = n_bursts
    denominator = 1.0 + z**2 / n
    center = (poi + z**2 / (2.0 * n)) / denominator
    half_width = (
        z
        * np.sqrt(poi * (1.0 - poi) / n + z**2 / (4.0 * n**2))
        / denominator
    )

    log(f"\n  burst bắt được : {n_hit}/{n_bursts}")
    log(
        f"  POI            : {poi * 100:.2f}% "
        f"[95% CI {(center - half_width) * 100:.2f}–{(center + half_width) * 100:.2f}%]"
    )

    return {
        "poi": poi,
        "ci": (center - half_width, center + half_width),
        "delay": delay,
        "N": n_bursts,
        "k": n_hit,
    }


# ═══ 7. PREDICT POI ══════════════════════════════════════════════════════════
def union_from_epochs(epoch_sorted: np.ndarray, burst_duration: float) -> float:
    """Độ dài hợp của [e*T_S-b, e*T_S+T_INT] cho các epoch thực sự có."""
    if len(epoch_sorted) == 0:
        return 0.0

    frame_times = epoch_sorted.astype(np.float64) * T_S
    lower = frame_times - burst_duration
    upper = frame_times + T_INT

    total = 0.0
    current_start = float(lower[0])
    current_end = float(upper[0])

    for interval_start, interval_end in zip(lower[1:], upper[1:]):
        interval_start = float(interval_start)
        interval_end = float(interval_end)

        if interval_start <= current_end:
            current_end = max(current_end, interval_end)
        else:
            total += current_end - current_start
            current_start = interval_start
            current_end = interval_end

    return total + current_end - current_start


def step7(
    timing: Dict[str, object],
    b_ms: float,
    loss_rate: float,
) -> Dict[str, float]:
    hdr("7 · DỰ ĐOÁN POI")

    burst_duration = b_ms / 1000.0
    unions_recorded = np.asarray([
        union_from_epochs(epoch_array, burst_duration)
        for epoch_array in timing["dwell_ep"]
    ])

    predicted_raw = float(np.mean(unions_recorded)) / float(timing["T_sweep"])
    predicted = min(1.0, max(0.0, predicted_raw))

    log(f"  E[|union|] recorded = {np.mean(unions_recorded) * 1e3:.3f} ms")
    log(f"  POI dự đoán         = {predicted * 100:.2f}%")

    if loss_rate > 0.0:
        unions_full = np.asarray([
            union_from_epochs(np.arange(int(n)), burst_duration)
            for n in timing["n_sched"]
        ])
        full_raw = float(np.mean(unions_full)) / float(timing["T_sweep"])
        full_prediction = min(1.0, max(0.0, full_raw))
        log(f"  POI nếu không mất frame = {full_prediction * 100:.2f}%")
    else:
        full_prediction = predicted

    continuous = np.nan
    ideal = np.nan

    if timing["ok_phase"]:
        raw_continuous = (
            float(timing["T_collect"]) + burst_duration
        ) / float(timing["T_sweep"])
        raw_ideal = raw_continuous / float(timing["d_collect"])

        continuous = min(1.0, max(0.0, raw_continuous))
        ideal = min(1.0, max(0.0, raw_ideal))

        log(f"\n  POI continuous model = {continuous * 100:.2f}%")
        log(f"  POI ideal collect    = {ideal * 100:.2f}%")

        if raw_continuous <= 1.0 and raw_ideal <= 1.0:
            log(
                f"  δ_collect × POI_ideal = "
                f"{timing['d_collect']:.4f} × {ideal * 100:.2f}% = "
                f"{timing['d_collect'] * ideal * 100:.2f}%"
            )
    else:
        log("\n  Không báo continuous/ideal POI vì phase validation không đạt.")

    return {
        "pred": predicted,
        "pred_no_loss": full_prediction,
        "cont": float(continuous),
        "ideal": float(ideal),
    }


# ═══ 8. CUMULATIVE POI ═══════════════════════════════════════════════════════
def step8(poi: float, t_sweep: float) -> None:
    hdr("8 · CUMULATIVE POI")

    log(f"  T_sweep = {t_sweep * 1e3:.2f} ms\n")
    log(f"    {'sweep':>6} | {'time':>10} | {'cumulative POI':>15}")
    log("    " + "-" * 39)

    for k in (1, 2, 5, 10, 20, 30, 50):
        cumulative = 1.0 - (1.0 - poi) ** k
        log(f"    {k:6d} | {k * t_sweep:8.3f} s | {cumulative * 100:13.1f}%")

    if poi <= 0.0:
        log("\n  POI = 0: không thể đạt target với mô hình hiện tại.")
        return

    if poi >= 1.0:
        log("\n  POI = 100%: đạt 90% và 99% sau 1 sweep.")
        return

    for target in (0.90, 0.99):
        k = int(np.ceil(np.log(1.0 - target) / np.log(1.0 - poi)))
        log(f"\n  đạt {target * 100:.0f}%: {k} sweep ≈ {k * t_sweep:.2f} s")


def validate_txoff_run(
    main_data: Dict[str, object],
    txoff_data: Dict[str, object],
    hop_burst: float,
) -> Tuple[float, float]:
    main_attrs = main_data["attrs"]
    tx_attrs = txoff_data["attrs"]

    for key in ("sample_rate", "fft_len", "db_min", "db_max"):
        if key in main_attrs and key in tx_attrs and main_attrs[key] != tx_attrs[key]:
            log(f"  ⚠ TX-off attr mismatch: {key}: {main_attrs[key]} vs {tx_attrs[key]}")

    tx_centers = np.asarray(txoff_data["center"])
    if not np.any(tx_centers == hop_burst):
        sys.exit("[FATAL] TX-off run không có hop burst")

    db_min = float(tx_attrs.get("db_min", -120.0))
    db_max = float(tx_attrs.get("db_max", 0.0))
    return db_min, db_max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--burst-csv", required=True)
    parser.add_argument("--f-burst", type=float, required=True)
    parser.add_argument("--tx-off-dir", default=None)
    parser.add_argument("--tol-ms", type=float, default=3.0)
    parser.add_argument("--threshold-db", type=float, default=None)
    args = parser.parse_args()

    tolerance = args.tol_ms / 1000.0
    if tolerance < 0:
        sys.exit("[FATAL] --tol-ms phải >= 0")

    t_on, t_off, b_ms = load_bursts(args.burst_csv)
    burst_duration = b_ms / 1000.0

    log(
        f"POI ANALYSIS v2 | b={b_ms:g} ms | {len(t_on)} burst | "
        f"f_burst={args.f_burst / 1e6:.3f} MHz"
    )

    data = load_h5(args.h5_dir)
    db_min, db_max, has_ep, timeline_ok = step1(data, t_on, t_off)
    
    if not has_ep:
        log("\n[STOP] không có epoch.")
        return

    if not timeline_ok:
        log("\n[STOP] timeline TX và RX không chồng lấn; không thể tính POI.")
        return

    loss_rate, duplicates = step2(data)
    timing = step3(data, args.f_burst, loss_rate, duplicates)

    center = np.asarray(data["center"])
    timestamps = np.asarray(data["ts"])
    fft = np.asarray(data["fft"])

    selected = center == timing["hop_burst"]
    burst_bin = bin_of(args.f_burst, float(timing["hop_burst"]))
    if not 0 <= burst_bin < FFT_LEN:
        sys.exit("[FATAL] burst nằm ngoài FFT grid của hop chính")

    snr_hop = snr_of(fft[selected], burst_bin, db_min, db_max)
    ts_hop = timestamps[selected]
    hdr("BURST-BIN SNR DIAGNOSTIC")

    for percentile in (50, 90, 95, 99, 99.5, 99.9, 100):
        log(
            f"  SNR p{percentile:>5}: "
            f"{np.percentile(snr_hop, percentile):+.2f} dB"
        )

    for level in (8, 10, 15, 20, 30, 40):
        count = int(np.count_nonzero(snr_hop > level))
        log(f"  frame > {level:2d} dB: {count:,}")

        # ── Diagnostic: các frame rất mạnh có phân bố giống burst 2 ms không? ──
    hdr("STRONG-FRAME STRUCTURE DIAGNOSTIC")

    epoch_all = np.asarray(data["epoch"])
    tune_all = np.asarray(data["tune"])

    epoch_hop = epoch_all[selected]
    tune_hop = tune_all[selected]

    strong_threshold = 20.0
    strong = snr_hop > strong_threshold
    strong_idx = np.flatnonzero(strong)

    log(f"  strong threshold : {strong_threshold:.1f} dB")
    log(f"  strong frames    : {len(strong_idx)}")

    # if len(strong_idx) == 0:
    #     log("\n[STOP] Không có strong frame để kiểm tra.")
    #     return

    strong_ts = ts_hop[strong_idx]
    strong_tune = tune_hop[strong_idx]
    strong_epoch = epoch_hop[strong_idx]
    strong_snr = snr_hop[strong_idx]

    # Số strong frame trong mỗi dwell/tune.
    unique_tunes, strong_per_tune = np.unique(
        strong_tune,
        return_counts=True,
    )

    count_values, count_occurrences = np.unique(
        strong_per_tune,
        return_counts=True,
    )

    log(f"  dwell có strong frame : {len(unique_tunes)}")
    log(
        "  strong frames/dwell  : "
        f"{dict(zip(count_values.tolist(), count_occurrences.tolist()))}"
    )

    log(
        f"  mean strong/dwell     : {strong_per_tune.mean():.3f}"
    )
    log(
        f"  max strong/dwell      : {strong_per_tune.max()}"
    )

    # Khoảng cách thời gian giữa các strong frame.
    order = np.argsort(strong_ts)
    sorted_ts = strong_ts[order]
    strong_gaps = np.diff(sorted_ts)

    if len(strong_gaps):
        log("\n  strong-frame timestamp gaps:")
        log(f"    median : {np.median(strong_gaps) * 1e3:.3f} ms")
        log(f"    p10    : {np.percentile(strong_gaps, 10) * 1e3:.3f} ms")
        log(f"    p90    : {np.percentile(strong_gaps, 90) * 1e3:.3f} ms")
        log(
            f"    gap ≈ T_s ({T_S * 1e3:.3f} ms): "
            f"{np.mean(np.abs(strong_gaps - T_S) < 0.2e-3) * 100:.1f}%"
        )

    # In các strong frame đầu tiên theo thời gian.
    log(
        f"\n  {'timestamp':>20} | {'tune':>10} | "
        f"{'epoch':>7} | {'SNR':>8}"
    )
    log("  " + "-" * 57)

    for idx in order[:30]:
        log(
            f"  {strong_ts[idx]:20.6f} | "
            f"{int(strong_tune[idx]):10d} | "
            f"{int(strong_epoch[idx]):7d} | "
            f"{strong_snr[idx]:7.2f}"
        )
        # ── Diagnostic: strong-event rate trước TX và trong TX run ─────────────
    hdr("PRE-TX INTERFERENCE CHECK")

    strong_threshold = 20.0
    time_guard = 1.0  # bỏ 1 giây quanh mép để tránh ảnh hưởng delay/buffer

    pre_start = float(ts_hop.min())
    pre_end = float(t_on[0] - time_guard)

    run_start = float(t_on[0] + time_guard)
    run_end = float(min(t_off[-1] - time_guard, ts_hop.max()))

    pre_mask = (
        (ts_hop >= pre_start)
        & (ts_hop < pre_end)
    )

    run_mask = (
        (ts_hop >= run_start)
        & (ts_hop <= run_end)
    )

    pre_duration = max(0.0, pre_end - pre_start)
    run_duration = max(0.0, run_end - run_start)

    pre_strong = int(np.count_nonzero(
        pre_mask & (snr_hop > strong_threshold)
    ))

    run_strong = int(np.count_nonzero(
        run_mask & (snr_hop > strong_threshold)
    ))

    pre_rate = (
        pre_strong / pre_duration
        if pre_duration > 0
        else np.nan
    )

    run_rate = (
        run_strong / run_duration
        if run_duration > 0
        else np.nan
    )

    log(f"  threshold          : {strong_threshold:.1f} dB")
    log(f"  pre-TX interval    : {pre_duration:.3f} s")
    log(f"  pre-TX strong      : {pre_strong}")
    log(f"  pre-TX rate        : {pre_rate:.4f} event/s")
    log(f"  TX-run interval    : {run_duration:.3f} s")
    log(f"  TX-run strong      : {run_strong}")
    log(f"  TX-run rate        : {run_rate:.4f} event/s")

    if np.isfinite(pre_rate) and pre_rate > 0:
        log(f"  run/pre rate ratio : {run_rate / pre_rate:.3f}×")

    if pre_strong > 0:
        log("\n  ⚠ Có strong frame trước TX → top-SNR frames không thể")
        log("    được dùng trực tiếp để estimate delay.")
    delay_info = estimate_delay_robust(snr_hop, ts_hop, t_on, t_off)
    hdr("ROBUST DELAY DIAGNOSTIC")

    log(
        f"  delay chính       : "
        f"{float(delay_info['delay']) * 1e3:+.3f} ms"
    )
    log(
        f"  delay thứ hai     : "
        f"{float(delay_info['second_delay']) * 1e3:+.3f} ms"
    )
    log(
        f"  burst associated  : "
        f"{delay_info['n_associated']}/{len(t_on)}"
    )
    log(
        f"  burst max SNR >8  : "
        f"{delay_info['hit8']} "
        f"(median delay sai: {delay_info['hit8_median']:.1f})"
    )
    log(
        f"  burst max SNR >10 : "
        f"{delay_info['hit10']} "
        f"(median delay sai: {delay_info['hit10_median']:.1f})"
    )
    log(
        f"  capped score      : "
        f"{delay_info['robust_score']:.2f}"
    )
    log(
        f"  second score      : "
        f"{delay_info['second_robust_score']:.2f}"
    )
    log(
        f"  peak ratio        : "
        f"{delay_info['peak_ratio']:.3f}"
    )
    log(
        f"  robust z-score    : "
        f"{delay_info['score_z']:.2f}"
    )
    log(
        f"  boundary          : "
        f"{delay_info['at_boundary']}"
    )
    log(
        f"  ambiguous         : "
        f"{delay_info['ambiguous']}"
    )

    if (
        delay_info["at_boundary"]
        or delay_info["ambiguous"]
    ):
        log(
            "\n[STOP] Robust delay chưa đủ rõ; "
            "chưa hiệu chuẩn threshold hoặc tính POI."
        )
        return
    delay = float(delay_info["delay"])
    hdr("4 · DELAY VALIDATION")
    log(
        f"  delay      : {delay * 1e3:+.3f} ms\n"
        f"  best score : {delay_info['peak_score']}/{delay_info['n_candidates']}\n"
        f"  second     : {delay_info['second_score']}\n"
        f"  boundary   : {delay_info['at_boundary']}\n"
        f"  ambiguous  : {delay_info['ambiguous']}"
    )

    minimum_peak = max(
        10,
        int(np.ceil(0.03 * delay_info["n_candidates"])),
    )

    enough_matches = delay_info["peak_score"] >= minimum_peak
    enough_separation = (
        delay_info["peak_score"] - delay_info["second_score"] >= 3
    )
    delay_ok = (
        enough_matches
        and enough_separation
        and not delay_info["at_boundary"]
        and not delay_info["ambiguous"]
    )

    log(f"  minimum peak : {minimum_peak}")
    log(f"  enough match : {enough_matches}")
    log(f"  separated    : {enough_separation}")

    if not delay_ok:
        log("\n[STOP] Delay chưa hợp lệ; không được tạo tập noise hoặc tính POI.")
        return

    txoff_snr: Optional[np.ndarray] = None
    if args.tx_off_dir:
        txoff_data = load_h5(args.tx_off_dir)
        tx_db_min, tx_db_max = validate_txoff_run(data, txoff_data, float(timing["hop_burst"]))
        tx_selected = np.asarray(txoff_data["center"]) == timing["hop_burst"]
        txoff_snr = snr_of(
            np.asarray(txoff_data["fft"])[tx_selected],
            burst_bin,
            tx_db_min,
            tx_db_max,
        )

    if args.threshold_db is not None:
        threshold = float(args.threshold_db)
    else:
        threshold_result = step5(
            snr_hop,
            ts_hop - delay,
            t_on,
            t_off,
            len(t_on),
            tolerance,
            burst_duration,
            txoff_snr,
        )
        if threshold_result is None:
            return
        threshold = threshold_result

    step4(
        data,
        args.f_burst,
        timing,
        db_min,
        db_max,
        threshold,
        t_on,
        t_off,
        delay,
        tolerance,
    )

    measured = step6(
        snr_hop,
        ts_hop,
        t_on,
        t_off,
        threshold,
        tolerance,
        delay_info,
    )

    predicted = step7(timing, b_ms, loss_rate)

    hdr("TÓM TẮT")
    log(
        f"  POI đo được : {measured['poi'] * 100:.2f}% "
        f"[95% CI {measured['ci'][0] * 100:.2f}–{measured['ci'][1] * 100:.2f}%]"
    )
    log(f"  POI dự đoán : {predicted['pred'] * 100:.2f}%")

    error_points = (measured["poi"] - predicted["pred"]) * 100.0
    log(f"  sai lệch    : {error_points:+.2f} điểm %")

    inside_ci = measured["ci"][0] <= predicted["pred"] <= measured["ci"][1]
    if inside_ci:
        log("  ✔ prediction nằm trong Wilson 95% CI của phép đo")
    else:
        log("  ⚠ prediction nằm ngoài Wilson 95% CI của phép đo")

    if 0.0 < measured["poi"] < 1.0:
        standard_error = np.sqrt(
            measured["poi"] * (1.0 - measured["poi"]) / measured["N"]
        )
        if standard_error > 0:
            log(f"  |error|/SE  : {abs(error_points / 100.0) / standard_error:.2f}")
    else:
        log("  Không báo z-score khi POI ở biên 0 hoặc 1; dùng Wilson CI.")

    step8(float(measured["poi"]), float(timing["T_sweep"]))


if __name__ == "__main__":
    main()
