#!/usr/bin/env python3
"""
RX-only controller using GNU Radio top_block (no GUI).

Flow:
- RX: uhd.usrp_source -> blocks.stream_to_vector(FFT_LEN) -> FrameProcessor
"""

import signal
import threading
import time
import math
import queue
from pathlib import Path
from typing import List, Tuple

import numpy as np
from gnuradio import blocks, gr, uhd


RX_SERIAL = "34D628E"
FPGA_BIN = "/home/datnguyen/Downloads/USRP_B210_FPGA_Thesis/uhd/usrp_b210_fpga_fixed_freq_shift_v1.bin"

RX_SAMP_RATE = 50e6
RX_GAIN_DB = 22
RX_START_HZ = 90e6
RX_STOP_HZ = 1e9
RX_STEP_HZ = 40e6
RX_INIT_HZ = 4e8
RX_WARMUP_HZ = 5e8
RX_WARMUP_DELAY_S = 0.1
RX_WARMUP_HOLD_S = 0.1
RX_FINAL_HZ = 5e8
FFT_LEN = 1024
SWEEP_ENABLE = True
SWEEP_DWELL_MS = 10
MEASURE_SWEEP_TIMING = True
SWEEP_TIMING_LOG_EVERY = 1

DETECT_MARGIN_DB = 5.0
MIN_SIGNAL_BINS = 2
MIN_PEAK_SNR_DB = 10.5
DC_GUARD_BINS = 24
DC_GUARD_HZ = 1.5e6

DETECT_EVERY = 5
LOG_FILE = Path(__file__).with_name("sweep_gr.log")
LOG_QUEUE_MAXSIZE = 10000
LOG_BATCH_SIZE = 200
LOG_FLUSH_INTERVAL_S = 0.5
# Drop vectors for a short window after each retune to avoid queued old-center frames.
SETTLE_TIME_AFTER_TUNE_MS = 0.2


stop_event = threading.Event()
log_queue = queue.Queue(maxsize=LOG_QUEUE_MAXSIZE)
log_drop_lock = threading.Lock()
log_drop_count = 0


def _enqueue_log_line(line: str):
    global log_drop_count
    try:
        log_queue.put_nowait(line)
    except queue.Full:
        with log_drop_lock:
            log_drop_count += 1


def _log_writer_worker():
    batch = []
    next_flush = time.perf_counter() + LOG_FLUSH_INTERVAL_S

    with LOG_FILE.open("a", encoding="utf-8") as f:
        while True:
            timeout_s = max(0.0, next_flush - time.perf_counter())
            try:
                item = log_queue.get(timeout=timeout_s)
            except queue.Empty:
                item = None

            if item is None:
                now = time.perf_counter()
                if batch and (now >= next_flush):
                    f.write("\n".join(batch) + "\n")
                    f.flush()
                    batch.clear()
                    next_flush = now + LOG_FLUSH_INTERVAL_S

                if stop_event.is_set() and log_queue.empty():
                    break
                continue

            batch.append(item)
            if len(batch) >= LOG_BATCH_SIZE:
                f.write("\n".join(batch) + "\n")
                f.flush()
                batch.clear()
                next_flush = time.perf_counter() + LOG_FLUSH_INTERVAL_S

        if batch:
            f.write("\n".join(batch) + "\n")

        with log_drop_lock:
            dropped = log_drop_count
        if dropped > 0:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] [LOG] dropped_lines={dropped} (queue full)\n")
        f.flush()


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    _enqueue_log_line(line)


def reset_log_file():
    # Truncate on startup so each run has a fresh log.
    LOG_FILE.write_text("", encoding="utf-8")


def compute_settle_frames() -> int:
    vectors_per_second = (RX_SAMP_RATE/2) / FFT_LEN
    settle_frames = int(math.ceil((SETTLE_TIME_AFTER_TUNE_MS / 1000.0) * vectors_per_second))
    return max(1, settle_frames)


def detect_signals(
    fft_complex: np.ndarray, center_freq_hz: float
) -> Tuple[float, float, List[dict], int, float]:
    fft_shifted = np.fft.fftshift(fft_complex)
    power_db = 10.0 * np.log10(np.abs(fft_shifted) ** 2 + 1e-12)
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(fft_complex.size, 1.0 / RX_SAMP_RATE))
    freq_axis = center_freq_hz + fft_freqs

    noise_floor = float(np.median(power_db))
    threshold = noise_floor + DETECT_MARGIN_DB

    global_argmax_bin = int(np.argmax(power_db))
    global_argmax_freq = float(freq_axis[global_argmax_bin])

    mask = power_db > threshold
    mid = fft_complex.size // 2
    lo = max(0, mid - DC_GUARD_BINS)
    hi = min(mask.size, mid + DC_GUARD_BINS + 1)
    mask[lo:hi] = False

    signals = []
    idx = 0
    bin_bw = RX_SAMP_RATE / fft_complex.size
    while idx < mask.size:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < mask.size and mask[idx]:
            idx += 1
        end = idx
        if (end - start) < MIN_SIGNAL_BINS:
            continue

        cluster_lo_hz = float(freq_axis[start])
        cluster_hi_hz = float(freq_axis[end - 1])
        dc_lo_hz = center_freq_hz - DC_GUARD_HZ
        dc_hi_hz = center_freq_hz + DC_GUARD_HZ
        if not (cluster_hi_hz < dc_lo_hz or cluster_lo_hz > dc_hi_hz):
            continue

        segment_freq = freq_axis[start:end]
        segment_power = power_db[start:end]
        peak_idx = start + int(np.argmax(segment_power))
        peak_db = float(power_db[peak_idx])
        peak_snr_db = peak_db - noise_floor
        if peak_snr_db < MIN_PEAK_SNR_DB:
            continue

        signals.append(
            {
                "peak_bin": int(peak_idx),
                "peak_db": peak_db,
                "peak_snr_db": peak_snr_db,
                "peak_freq": float(freq_axis[peak_idx]),
                "width_hz": float((end - start) * bin_bw),
                "span_start": float(segment_freq[0]),
                "span_end": float(segment_freq[-1]),
            }
        )

    return noise_floor, threshold, signals, global_argmax_bin, global_argmax_freq


class FrameProcessor(gr.sync_block):
    """Consume vectorized complex FFT frames and run the same detector as dual_usrp_sweep."""

    def __init__(self, fft_len: int, center_hz: float, detect_every: int):
        gr.sync_block.__init__(
            self,
            name="FrameProcessor",
            in_sig=[(np.complex64, fft_len)],
            out_sig=None,
        )
        self.fft_len = int(fft_len)
        self.center_hz = float(center_hz)
        self.detect_every = max(1, int(detect_every))
        self.frame_counter = 0
        self.tune_id = 0
        self.frame_epoch = 0
        self.settle_frames = 0
        self._lock = threading.Lock()

    def begin_tune(self, tune_id: int, freq_hz: float, settle_frames: int):
        with self._lock:
            self.tune_id = int(tune_id)
            self.center_hz = float(freq_hz)
            self.frame_epoch = 0
            self.settle_frames = max(0, int(settle_frames))

    def work(self, input_items, output_items):
        frames = input_items[0]
        for i in range(len(frames)):
            vec = np.asarray(frames[i], dtype=np.complex64)
            self.frame_counter += 1

            with self._lock:
                tune_id = self.tune_id
                center_hz = self.center_hz
                if self.settle_frames > 0:
                    self.settle_frames -= 1
                    self.frame_epoch += 1
                    continue
                frame_epoch = self.frame_epoch
                self.frame_epoch += 1

            if frame_epoch % self.detect_every != 0:
                continue

            noise_floor, threshold, signals, g_bin, g_freq = detect_signals(vec, center_hz)

            if signals:
                for sig in signals:
                    log(
                        "[SIG] tune_id={} | frame_epoch={} | F={:.1f} MHz | peak_bin={} | peak={:+.1f} dB | snr={:.1f} dB | "
                        "freq={:.3f} MHz | width={:.1f} kHz | noise={:+.1f} dB | thr={:+.1f} dB | "
                        "g_argmax_bin={} | g_argmax_freq={:.3f} MHz".format(
                            tune_id,
                            frame_epoch,
                            center_hz / 1e6,
                            sig["peak_bin"],
                            sig["peak_db"],
                            sig["peak_snr_db"],
                            sig["peak_freq"] / 1e6,
                            sig["width_hz"] / 1e3,
                            noise_floor,
                            threshold,
                            g_bin,
                            g_freq / 1e6,
                        )
                    )

        return len(frames)


class RxTopBlock(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "dual_usrp_sweep_gr_rx", catch_exceptions=True)

        self.center_hz = RX_INIT_HZ

        self.usrp_source = uhd.usrp_source(
            ",".join(
                (
                    f"serial={RX_SERIAL}",
                    f"fpga={FPGA_BIN}",
                    "num_recv_frames=2048",
                    "recv_frame_size=8192",
                    f"spp={FFT_LEN}",
                )
            ),
            uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc16",
                args="",
                channels=list(range(0, 1)),
            ),
        )
        self.usrp_source.set_samp_rate(RX_SAMP_RATE)
        self.usrp_source.set_center_freq(self.center_hz, 0)
        self.usrp_source.set_antenna("RX2", 0)
        self.usrp_source.set_gain(RX_GAIN_DB, 0)
        self.usrp_source.set_auto_dc_offset(True, 0)
        self.usrp_source.set_auto_iq_balance(True, 0)
        self.usrp_source.set_rx_agc(False, 0)

        self.stream_to_vector = blocks.stream_to_vector(gr.sizeof_gr_complex * 1, FFT_LEN)
        self.frame_processor = FrameProcessor(FFT_LEN, self.center_hz, DETECT_EVERY)

        self.connect((self.usrp_source, 0), (self.stream_to_vector, 0))
        self.connect((self.stream_to_vector, 0), (self.frame_processor, 0))

        self.tune_id = 0
        self.set_center_freq(self.center_hz)

    def set_center_freq(self, freq_hz: float):
        new_freq = float(freq_hz)
        if self.center_hz == new_freq:
            return

        self.center_hz = new_freq
        self.tune_id += 1
        settle_frames = compute_settle_frames()
        self.usrp_source.set_center_freq(self.center_hz, 0)
        self.frame_processor.begin_tune(self.tune_id, self.center_hz, settle_frames)
        log(
            f"[RX] tune_id={self.tune_id} -> {self.center_hz/1e6:.1f} MHz "
            f"(settle_frames={settle_frames}, settle_ms={SETTLE_TIME_AFTER_TUNE_MS:.1f})"
        )


def run_warmup_and_optional_sweep(tb: RxTopBlock):
    # Match signal_detection warm-up sequence: 400 -> (0.5s) 900 -> (0.2s) 500
    time.sleep(RX_WARMUP_DELAY_S)
    if stop_event.is_set():
        return

    tb.set_center_freq(RX_WARMUP_HZ)
    time.sleep(RX_WARMUP_HOLD_S)
    if stop_event.is_set():
        return

    tb.set_center_freq(RX_FINAL_HZ)

    if not SWEEP_ENABLE:
        return

    dwell_s = SWEEP_DWELL_MS / 1000.0
    center_hz = RX_FINAL_HZ
    loop_idx = 0
    prev_tune_ts = time.perf_counter()
    while not stop_event.is_set():
        tune_start = time.perf_counter()
        tb.set_center_freq(center_hz)
        tune_end = time.perf_counter()

        # Sleep full dwell AFTER tune so FrameProcessor gets valid data
        time.sleep(dwell_s)
        sleep_end = time.perf_counter()

        center_hz += RX_STEP_HZ
        if center_hz > RX_STOP_HZ:
            center_hz = RX_START_HZ

        loop_idx += 1
        if MEASURE_SWEEP_TIMING and (loop_idx % max(1, SWEEP_TIMING_LOG_EVERY) == 0):
            sleep_ms = (sleep_end - tune_end) * 1000.0
            tune_ms = (tune_end - tune_start) * 1000.0
            interval_ms = (tune_start - prev_tune_ts) * 1000.0
            log(
                "[TIMING] loop={} | target={:.3f} ms | sleep={:.3f} ms | tune_call={:.3f} ms | "
                "interval_between_tunes={:.3f} ms".format(
                    loop_idx,
                    SWEEP_DWELL_MS,
                    sleep_ms,
                    tune_ms,
                    interval_ms,
                )
            )

        prev_tune_ts = tune_start


def main():
    reset_log_file()
    tb = RxTopBlock()
    log_thread = threading.Thread(target=_log_writer_worker, daemon=True)

    def _sig_handler(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    log_thread.start()
    tb.start()

    ctrl_thread = threading.Thread(target=run_warmup_and_optional_sweep, args=(tb,), daemon=True)
    ctrl_thread.start()

    while not stop_event.is_set():
        time.sleep(0.2)

    tb.stop()
    tb.wait()
    ctrl_thread.join(timeout=1.0)
    log_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()