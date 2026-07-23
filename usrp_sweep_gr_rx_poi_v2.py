#!/usr/bin/env python3
"""
RX-only controller using GNU Radio top_block (no GUI).

Flow:
- RX: uhd.usrp_source -> blocks.stream_to_vector(FFT_LEN) -> FrameProcessor

Streaming:
- FrameProcessor writes detected signals + quantized FFT into two shared-memory
  SPSC ring buffers.
- Two separate processes (signal_publisher, fft_publisher) read from the ring
  buffers and publish via ZMQ PUB sockets.
- Main process (GNU Radio + detection) pinned to cores 0-1.
- Signal publisher pinned to core 3, FFT publisher pinned to core 2.
"""

import atexit
import argparse
import json
import multiprocessing
import os
import signal as signal_mod
import struct
import threading
import time
import math
import queue
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
from gnuradio import blocks, gr, uhd
import pmt

from shm_ring_buffer import ShmRingBuffer
import psutil
import zmq


RX_SERIAL = "34D628E"
FPGA_BIN = "/home/datnguyen/USRP_Spectrum_Analyzer_Thesis/b200_fixed_18h_1905.bin"

RX_SAMP_RATE 	= 50e6
RX_BW 		= 30e6
RX_GAIN_DB 	= 15
RX_START_HZ 	= 358e6
RX_STOP_HZ 	= 508e6
RX_STEP_HZ 	= 25e6
RX_INIT_HZ 	= 400e6
RX_WARMUP_HZ 	= 500e6
RX_FINAL_HZ 	= 358e6
RX_WARMUP_DELAY_S = 1
RX_WARMUP_HOLD_S  = 1
FFT_LEN = 2048
SWEEP_ENABLE = True
SWEEP_DWELL_MS = 15
MEASURE_SWEEP_TIMING = True
SWEEP_TIMING_LOG_EVERY = 1

DETECT_MARGIN_DB = 4.0
# MIN_SIGNAL_BINS = 2
MIN_SIGNAL_BINS = 4
MIN_PEAK_SNR_DB = 10
# EDGE_GUARD_BINS = 20
# DC_GUARD_BINS = 24
EDGE_GUARD_BINS = 40
DC_GUARD_BINS = 48
DC_GUARD_HZ = 1.5e6

DETECT_EVERY = 5
LOG_FILE = Path(__file__).resolve().with_name("sweep_gr.log")
DEBUG_FRAME_DIR = Path(__file__).resolve().with_name("debug_frames")
DEBUG_FRAME_INDEX = 0
LOG_QUEUE_MAXSIZE = 10000
LOG_BATCH_SIZE = 200
LOG_FLUSH_INTERVAL_S = 0.5
# Timestamp/stale-frame handling.  UHD emits rx_time/rx_freq tags at stream
# start and after a direct set_center_freq() call.  The custom FPGA keeps one
# FFT frame out of every 50 raw FFT frames.
FPGA_FRAME_DECIM = 50
RX_FRAME_PERIOD_S = FFT_LEN * FPGA_FRAME_DECIM / RX_SAMP_RATE
POST_TUNE_DISCARD_FRAMES = 1   # additionally discard one complete vector after tag boundary
TUNE_TAG_TIMEOUT_S = 0.5
RX_FREQ_TAG_TOLERANCE_HZ = 2e3
TUNE_CAPTURE_GUARD_S = 0.0002

# ── ZMQ streaming configuration ──────────────────────────────────────────────
ZMQ_SIGNAL_PORT = 5555
ZMQ_FFT_PORT = 5556
ZMQ_SIGNAL_HWM = 500
ZMQ_FFT_HWM = 200
ZMQ_TELEM_PORT = 5557
ZMQ_CONTROL_PORT = 5558

# Shared-memory ring buffer sizing
SHM_SIG_NAME = "spec_sig_ring"
SHM_FFT_NAME = "spec_fft_ring"
SHM_SIG_SLOTS = 512
SHM_SIG_SLOT_SIZE = 16384      # max JSON payload bytes per signal event
SHM_FFT_SLOTS = 256
SHM_FFT_SLOT_SIZE = 2048 + 768  # 2048 uint8 bins + timestamp-rich JSON metadata

# FFT quantization
FFT_QUANTIZE_DB_MIN = -120.0
FFT_QUANTIZE_DB_MAX = 0.0

# Send every Nth FFT frame to ZMQ (bandwidth reduction)
FFT_STREAM_DECIMATION = 1

# CPU core pinning (Pi 5 has cores 0-3)
CORES_MAIN = {0, 1}   # GNU Radio + detection
CORE_FFT_PUB = {2}    # FFT ZMQ publisher
CORE_SIG_PUB = {3}    # Signal ZMQ publisher


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


def save_debug_frame(
    frame_vec: np.ndarray, tune_id: int, center_hz: float, frame_epoch: int
):
    DEBUG_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    center_mhz = center_hz / 1e6
    file_name = (
        f"tune_{tune_id:04d}_center_{center_mhz:.1f}MHz_frame_{frame_epoch + 1:02d}.npz"
    )
    file_path = DEBUG_FRAME_DIR / file_name
    np.savez(
        file_path,
        frame=frame_vec.astype(np.complex64, copy=False),
        tune_id=np.int32(tune_id),
        center_hz=np.float64(center_hz),
        frame_epoch=np.int32(frame_epoch),
        sample_rate=np.float64(RX_SAMP_RATE),
        fft_len=np.int32(frame_vec.size),
    )
    log(f"[DEBUG] saved_frame={file_path.name}")


def _pmt_rx_time_to_seconds(value) -> float | None:
    """Convert UHD rx_time PMT tuple (whole seconds, fractional seconds)."""
    try:
        if not pmt.is_tuple(value) or pmt.length(value) < 2:
            return None
        whole = int(pmt.to_uint64(pmt.tuple_ref(value, 0)))
        frac = float(pmt.to_double(pmt.tuple_ref(value, 1)))
        return float(whole) + frac
    except Exception:
        return None


def _pmt_number_to_float(value) -> float | None:
    try:
        return float(pmt.to_double(value))
    except Exception:
        try:
            return float(pmt.to_python(value))
        except Exception:
            return None


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
    if EDGE_GUARD_BINS > 0:
        mask[:EDGE_GUARD_BINS] = False
        mask[-EDGE_GUARD_BINS:] = False
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


def quantize_fft_to_uint8(fft_complex: np.ndarray) -> np.ndarray:
    """Convert complex FFT vector to fftshift'd magnitude in dB, quantized to uint8."""
    mag_sq = np.abs(np.fft.fftshift(fft_complex)) ** 2
    power_db = 10.0 * np.log10(mag_sq + 1e-12)
    clipped = np.clip(power_db, FFT_QUANTIZE_DB_MIN, FFT_QUANTIZE_DB_MAX)
    scaled = (clipped - FFT_QUANTIZE_DB_MIN) / (FFT_QUANTIZE_DB_MAX - FFT_QUANTIZE_DB_MIN) * 255.0
    return scaled.astype(np.uint8)


# ── ZMQ publisher processes ──────────────────────────────────────────────────

def _signal_publisher_process(shm_name, slot_count, slot_size, port, hwm, bind_ip="0.0.0.0"):
    """Publish detected-signal JSON over ZMQ PUB (one process, one core)."""
    import zmq

    try:
        os.sched_setaffinity(0, CORE_SIG_PUB)
    except OSError:
        pass  # non-Linux or insufficient permissions

    ring = ShmRingBuffer(shm_name, slot_count, slot_size, create=False)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, hwm)
    sock.bind(f"tcp://{bind_ip}:{port}")

    while True:
        data = ring.read()
        if data is None:
            time.sleep(0.0005)
            continue
        try:
            sock.send(data, zmq.NOBLOCK)
        except zmq.Again:
            pass  # subscriber too slow — drop


def _fft_publisher_process(shm_name, slot_count, slot_size, port, hwm, bind_ip="0.0.0.0"):
    """Publish quantized FFT as ZMQ multipart [metadata_json, uint8_binary]."""
    import zmq

    try:
        os.sched_setaffinity(0, CORE_FFT_PUB)
    except OSError:
        pass

    ring = ShmRingBuffer(shm_name, slot_count, slot_size, create=False)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, hwm)
    sock.bind(f"tcp://{bind_ip}:{port}")

    while True:
        data = ring.read()
        if data is None:
            time.sleep(0.0005)
            continue
        # Slot format: [2-byte meta_len][meta_json_bytes][fft_uint8_bytes]
        if len(data) < 2:
            continue
        meta_len = struct.unpack_from("<H", data, 0)[0]
        if len(data) < 2 + meta_len:
            continue
        meta_bytes = data[2: 2 + meta_len]
        fft_bytes = data[2 + meta_len:]
        try:
            sock.send_multipart([meta_bytes, fft_bytes], zmq.NOBLOCK)
        except zmq.Again:
            pass


def _telemetry_publisher_thread(tb: 'RxTopBlock', port: int = ZMQ_TELEM_PORT, interval_s: float = 1.0, bind_ip: str = "0.0.0.0"):
    """Publish system + app telemetry as JSON over ZMQ PUB.

    Runs in main process (reads `tb` attributes for USRP state).
    """
    try:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.SNDHWM, 100)
        sock.bind(f"tcp://{bind_ip}:{port}")
    except Exception:
        return

    def read_temp_c():
        # Try psutil sensors first
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for k in temps:
                    if temps[k]:
                        return float(temps[k][0].current)
        except Exception:
            pass
        # Fallback to sysfs
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                v = int(f.read().strip())
                return float(v) / 1000.0
        except Exception:
            return None

    start_time = time.time()
    while not stop_event.is_set():
        ts = datetime.now(timezone.utc).isoformat()
        cpu_pct = None
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
        except Exception:
            cpu_pct = None
        try:
            vm = psutil.virtual_memory()
            ram_pct = float(vm.percent)
            ram_used_mb = float(vm.used) / (1024 * 1024)
        except Exception:
            ram_pct = None
            ram_used_mb = None

        temp_c = read_temp_c()
        uptime_s = time.time() - start_time

        # Application-level state from top block
        app_state = {}
        try:
            app_state["center_hz"] = float(tb.center_hz)
            app_state["tune_id"] = int(tb.tune_id)
            app_state["gain_db"] = float(getattr(tb, "current_gain", RX_GAIN_DB))
            fp = tb.frame_processor
            app_state["timestamp_source"] = "uhd_rx_time"
            app_state["stale_frames_discarded"] = int(fp.stale_frames_discarded)
            app_state["timestamp_missing_frames"] = int(fp.timestamp_missing_frames)
            app_state["timestamp_nonmonotonic_frames"] = int(fp.timestamp_nonmonotonic_frames)
            app_state["tune_tag_timeouts"] = int(fp.tune_tag_timeouts)
            app_state["usrp_host_time_offset_ms"] = float(tb.device_host_offset_s * 1e3)
            app_state["usrp_time_read_rtt_ms"] = float(tb.device_time_read_rtt_s * 1e3)
        except Exception:
            pass

        payload = {
            "ts": ts,
            "temp_c": temp_c,
            "cpu_percent": cpu_pct,
            "ram_percent": ram_pct,
            "ram_used_mb": ram_used_mb,
            "uptime_s": uptime_s,
            "app": app_state,
        }

        try:
            sock.send_json(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            # subscriber too slow; drop
            pass

        time.sleep(interval_s)


def _control_listener_thread(tb: 'RxTopBlock', port: int = ZMQ_CONTROL_PORT, bind_ip: str = "0.0.0.0"):
    """Listen for control commands and update USRP gain at runtime."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.RCVHWM, 10)
    sock.bind(f"tcp://{bind_ip}:{port}")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    try:
        while not stop_event.is_set():
            events = dict(poller.poll(timeout=200))
            if sock not in events:
                continue

            raw = sock.recv()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                sock.send_json({"status": "error", "msg": "Invalid JSON"})
                continue

            if not isinstance(payload, dict):
                sock.send_json({"status": "error", "msg": "Invalid format"})
                continue

            action = payload.get("action")
            value = payload.get("value")
            if action != "set_gain":
                sock.send_json({"status": "error", "msg": "Invalid action or format"})
                continue

            try:
                gain_db = float(value)
            except (TypeError, ValueError):
                sock.send_json({"status": "error", "msg": "Invalid gain value"})
                continue

            tb.set_gain(gain_db)
            sock.send_json({"status": "success", "msg": f"Gain updated to {gain_db} dB"})
    finally:
        sock.close()
        ctx.term()


class FrameProcessor(gr.sync_block):
    """Consume FFT vectors, preserve UHD acquisition time, and reject stale retune data."""

    def __init__(self, fft_len: int, center_hz: float, detect_every: int,
                 sig_ring: ShmRingBuffer = None, fft_ring: ShmRingBuffer = None):
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

        # Start fail-closed: no frame is published until a matching rx_freq/rx_time
        # boundary has been seen for the requested tune.
        self._discard_all = True
        self._awaiting_tune_boundary = False
        self._expected_freq_hz = None
        self._minimum_rx_time = float("-inf")
        self._matching_freq_tag_seen = False
        self._matching_freq_tag_item = None
        self._post_tune_discard_remaining = 0
        self._settled = threading.Event()

        # Acquisition-time reconstruction. rx_time gives an exact hardware-time
        # anchor; subsequent kept FFT vectors are spaced by RX_FRAME_PERIOD_S.
        self._time_anchor_item = None
        self._time_anchor_secs = None
        self._last_acq_time = None

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._detect_busy = False
        self._sig_ring = sig_ring
        self._fft_ring = fft_ring

        self._rx_time_key = pmt.intern("rx_time")
        self._rx_freq_key = pmt.intern("rx_freq")

        # Diagnostic counters exposed through telemetry.
        self.stale_frames_discarded = 0
        self.timestamp_missing_frames = 0
        self.timestamp_nonmonotonic_frames = 0
        self.tune_tag_timeouts = 0
        self._discarded_this_tune = 0

    def begin_tune(self, tune_id: int, freq_hz: float):
        """Arm a new tune and reject all queued data until its UHD tags arrive."""
        with self._cv:
            while self._detect_busy:
                self._cv.wait(timeout=0.01)
            self.tune_id = int(tune_id)
            self.center_hz = float(freq_hz)
            self.frame_epoch = 0
            self._discard_all = True
            self._awaiting_tune_boundary = True
            self._expected_freq_hz = float(freq_hz)
            self._minimum_rx_time = float("inf")
            self._matching_freq_tag_seen = False
            self._matching_freq_tag_item = None
            self._post_tune_discard_remaining = 0
            self._discarded_this_tune = 0
            self._settled.clear()

    def complete_tune(self, minimum_rx_time: float):
        """Publish the first acceptable hardware time after set_center_freq returns."""
        with self._lock:
            self._minimum_rx_time = float(minimum_rx_time)

    def wait_until_settled(self, timeout_s: float) -> bool:
        ok = self._settled.wait(timeout=max(0.0, float(timeout_s)))
        if not ok:
            with self._lock:
                self.tune_tag_timeouts += 1
        return ok

    def is_settled(self) -> bool:
        return self._settled.is_set()

    def _process_tags_for_item(self, abs_item: int, item_tags) -> bool:
        """Update time/frequency tag state. Return True for the tune-boundary frame."""
        rx_time = None
        rx_freq = None

        for tag in item_tags:
            try:
                key = pmt.symbol_to_string(tag.key)
            except Exception:
                continue
            if key == "rx_time":
                rx_time = _pmt_rx_time_to_seconds(tag.value)
            elif key == "rx_freq":
                rx_freq = _pmt_number_to_float(tag.value)

        if rx_time is not None:
            self._time_anchor_item = int(abs_item)
            self._time_anchor_secs = float(rx_time)

        if not self._awaiting_tune_boundary:
            return False

        if rx_freq is not None and self._expected_freq_hz is not None:
            if abs(rx_freq - self._expected_freq_hz) <= RX_FREQ_TAG_TOLERANCE_HZ:
                self._matching_freq_tag_seen = True
                self._matching_freq_tag_item = int(abs_item)

        # set_center_freq() may set _tag_now while old packets are still queued.
        # Therefore the new rx_freq tag alone is NOT sufficient: it can be attached
        # to stale data.  Reconstruct the hardware acquisition time of the current
        # vector and wait until it is newer than the time at which tuning completed.
        current_acq_time = self._acquisition_time_for_item(abs_item)
        freq_boundary_reached = (
            self._matching_freq_tag_seen
            and self._matching_freq_tag_item is not None
            and abs_item >= self._matching_freq_tag_item
        )
        time_boundary_valid = (
            current_acq_time is not None
            and current_acq_time >= self._minimum_rx_time
        )

        if not (freq_boundary_reached and time_boundary_valid):
            return False

        self._awaiting_tune_boundary = False
        self._post_tune_discard_remaining = POST_TUNE_DISCARD_FRAMES
        return True

    def _acquisition_time_for_item(self, abs_item: int) -> float | None:
        if self._time_anchor_item is None or self._time_anchor_secs is None:
            return None
        delta_frames = int(abs_item) - int(self._time_anchor_item)
        return float(self._time_anchor_secs) + delta_frames * RX_FRAME_PERIOD_S

    def work(self, input_items, output_items):
        frames = input_items[0]
        base_abs_item = int(self.nitems_read(0))

        tags_by_offset = {}
        try:
            for tag in self.get_tags_in_window(0, 0, len(frames)):
                tags_by_offset.setdefault(int(tag.offset), []).append(tag)
        except Exception as e:
            log(f"[TAG][ERROR] get_tags_in_window failed: {e}")

        for i in range(len(frames)):
            abs_item = base_abs_item + i
            vec = np.asarray(frames[i], dtype=np.complex64)
            self.frame_counter += 1
            should_detect = False
            drop_frame = False

            with self._lock:
                boundary_frame = self._process_tags_for_item(
                    abs_item, tags_by_offset.get(abs_item, ())
                )

                # Drop the first vector whose acquisition time crosses the clean
                # post-tune boundary. It may still straddle the hardware transition.
                if boundary_frame:
                    self.stale_frames_discarded += 1
                    self._discarded_this_tune += 1
                    drop_frame = True

                elif self._awaiting_tune_boundary:
                    self.stale_frames_discarded += 1
                    self._discarded_this_tune += 1
                    drop_frame = True

                elif self._post_tune_discard_remaining > 0:
                    self._post_tune_discard_remaining -= 1
                    self.stale_frames_discarded += 1
                    self._discarded_this_tune += 1
                    drop_frame = True
                    if self._post_tune_discard_remaining == 0:
                        self._discard_all = False
                        self._settled.set()
                        log(
                            f"[RX] tune_id={self.tune_id} clean boundary ready | "
                            f"discarded_vectors={self._discarded_this_tune}"
                        )

                elif self._discard_all:
                    drop_frame = True

                if drop_frame:
                    continue

                tune_id = self.tune_id
                center_hz = self.center_hz
                frame_epoch = self.frame_epoch
                self.frame_epoch += 1
                acq_time = self._acquisition_time_for_item(abs_item)

                if acq_time is None:
                    self.timestamp_missing_frames += 1
                    continue
                if self._last_acq_time is not None and acq_time <= self._last_acq_time:
                    self.timestamp_nonmonotonic_frames += 1
                    log(
                        f"[TAG][WARN] non-monotonic rx_time: current={acq_time:.9f} "
                        f"previous={self._last_acq_time:.9f}"
                    )
                    continue
                self._last_acq_time = acq_time

                if frame_epoch + 1 == DEBUG_FRAME_INDEX:
                    save_debug_frame(vec, tune_id, center_hz, frame_epoch)

                if frame_epoch % self.detect_every == 0:
                    self._detect_busy = True
                    should_detect = True

            process_time = time.time()
            acq_iso = datetime.fromtimestamp(acq_time, timezone.utc).isoformat()
            process_iso = datetime.fromtimestamp(process_time, timezone.utc).isoformat()

            # ── Stream quantized FFT to ring buffer ──────────────────
            if self._fft_ring is not None and frame_epoch % FFT_STREAM_DECIMATION == 0:
                fft_uint8 = quantize_fft_to_uint8(vec)
                meta = json.dumps({
                    "ts": acq_iso,                  # HDF5 timestamp = RF acquisition time
                    "ts_epoch": acq_time,
                    "process_ts": process_iso,     # separate host processing timestamp
                    "timestamp_source": "uhd_rx_time",
                    "tune_id": tune_id,
                    "center_hz": center_hz,
                    "sample_rate_hz": RX_SAMP_RATE,
                    "analog_bandwidth_hz": RX_BW,
                    "fft_len": self.fft_len,
                    "frame_period_s": RX_FRAME_PERIOD_S,
                    "db_min": FFT_QUANTIZE_DB_MIN,
                    "db_max": FFT_QUANTIZE_DB_MAX,
                    "epoch": frame_epoch,
                }, separators=(",", ":")).encode("utf-8")
                slot_data = struct.pack("<H", len(meta)) + meta + fft_uint8.tobytes()
                self._fft_ring.write(slot_data)

            if not should_detect:
                continue

            try:
                noise_floor, threshold, signals, g_bin, g_freq = detect_signals(vec, center_hz)

                with self._cv:
                    # Retune waits until this detect section exits.
                    if not self._discard_all and self.tune_id == tune_id and signals:
                        if self._sig_ring is not None:
                            sig_payload = json.dumps({
                                "ts": acq_iso,
                                "ts_epoch": acq_time,
                                "process_ts": process_iso,
                                "timestamp_source": "uhd_rx_time",
                                "tune_id": tune_id,
                                "center_hz": center_hz,
                                "frame_epoch": frame_epoch,
                                "noise_floor_db": noise_floor,
                                "threshold_db": threshold,
                                "signals": signals,
                            }, separators=(",", ":")).encode("utf-8")
                            self._sig_ring.write(sig_payload)

                        for sig in signals:
                            log(
                                "[SIG] tune_id={} | frame_epoch={} | t_acq={:.6f} | F={:.1f} MHz | "
                                "peak_bin={} | peak={:+.1f} dB | snr={:.1f} dB | freq={:.3f} MHz | "
                                "width={:.1f} kHz | noise={:+.1f} dB | thr={:+.1f} dB | "
                                "g_argmax_bin={} | g_argmax_freq={:.3f} MHz".format(
                                    tune_id,
                                    frame_epoch,
                                    acq_time,
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
            finally:
                with self._cv:
                    self._detect_busy = False
                    self._cv.notify_all()

        return len(frames)


class RxTopBlock(gr.top_block):
    def __init__(self, sig_ring=None, fft_ring=None):
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
        self.usrp_source.set_bandwidth(RX_BW, 0)
        self.current_gain = float(RX_GAIN_DB)
        self.usrp_source.set_auto_dc_offset(True, 0)
        self.usrp_source.set_auto_iq_balance(True, 0)
        self.usrp_source.set_rx_agc(False, 0)
        #self.usrp_source.set_bandwidth(8e6, 0)

        # Map B210 hardware time to the Pi's PTP-disciplined CLOCK_REALTIME.
        # Repeated set/readback compensates most USB control-path delay.  The final
        # measured residual is logged and must still be checked for short-burst POI.
        try:
            residual_s = 0.0
            for _ in range(3):
                target = time.time() - residual_s
                try:
                    self.usrp_source.set_time_now(uhd.time_spec(target), 0)
                except TypeError:
                    self.usrp_source.set_time_now(uhd.time_spec(target))
                time.sleep(0.01)
                t0 = time.time()
                device_now = self.usrp_source.get_time_now(0).get_real_secs()
                t1 = time.time()
                residual_s = float(device_now - 0.5 * (t0 + t1))

            residual_samples = []
            rtt_samples = []
            for _ in range(5):
                t0 = time.time()
                device_now = self.usrp_source.get_time_now(0).get_real_secs()
                t1 = time.time()
                residual_samples.append(float(device_now - 0.5 * (t0 + t1)))
                rtt_samples.append(float(t1 - t0))
            self.device_host_offset_s = float(np.median(residual_samples))
            self.device_time_read_rtt_s = float(np.median(rtt_samples))
            log(
                f"[TIME] RX USRP-host residual={self.device_host_offset_s*1e3:+.3f} ms | "
                f"median_read_RTT={self.device_time_read_rtt_s*1e3:.3f} ms"
            )
        except Exception as e:
            raise RuntimeError(f"Cannot synchronize/read RX USRP hardware time: {e}") from e

        self.stream_to_vector = blocks.stream_to_vector(gr.sizeof_gr_complex * 1, FFT_LEN)
        self.frame_processor = FrameProcessor(
            FFT_LEN, self.center_hz, DETECT_EVERY,
            sig_ring=sig_ring, fft_ring=fft_ring,
        )

        self.connect((self.usrp_source, 0), (self.stream_to_vector, 0))
        self.connect((self.stream_to_vector, 0), (self.frame_processor, 0))

        self.tune_id = 0
        self.set_center_freq(self.center_hz)

    def _device_time_now(self) -> float:
        try:
            return float(self.usrp_source.get_time_now(0).get_real_secs())
        except TypeError:
            return float(self.usrp_source.get_time_now().get_real_secs())

    def set_center_freq(self, freq_hz: float, force: bool = False) -> bool:
        new_freq = float(freq_hz)
        if not force and self.center_hz == new_freq:
            return False

        self.center_hz = new_freq
        self.tune_id += 1

        # Reject queued vectors before issuing the tune. A new rx_freq/rx_time tag
        # will be observed downstream, but acceptance additionally requires the
        # reconstructed acquisition time to be newer than tune completion.
        self.frame_processor.begin_tune(self.tune_id, self.center_hz)
        self.usrp_source.set_center_freq(self.center_hz, 0)
        tune_done_device_time = self._device_time_now()
        self.frame_processor.complete_tune(
            tune_done_device_time + TUNE_CAPTURE_GUARD_S
        )
        actual_freq = float(self.usrp_source.get_center_freq(0))
        log(
            f"[RX] tune_id={self.tune_id} requested={self.center_hz/1e6:.6f} MHz "
            f"actual={actual_freq/1e6:.6f} MHz | waiting for rx_freq/rx_time boundary"
        )
        return True

    def set_gain(self, gain_db: float):
        try:
            self.usrp_source.set_gain(float(gain_db), 0)
            self.current_gain = float(gain_db)
            log(f"[RX] set_gain={self.current_gain} dB")
        except Exception as e:
            log(f"[RX] set_gain_failed: {e}")


def run_warmup_and_optional_sweep(tb: RxTopBlock):
    time.sleep(RX_WARMUP_DELAY_S)
    if stop_event.is_set():
        return

    tb.set_center_freq(RX_WARMUP_HZ)
    # Warm-up time is still host-controlled; stale vectors are rejected by tags.
    time.sleep(RX_WARMUP_HOLD_S)
    if stop_event.is_set():
        return

    tb.set_center_freq(RX_FINAL_HZ)
    if not tb.frame_processor.wait_until_settled(TUNE_TAG_TIMEOUT_S):
        log(
            f"[RX][ERROR] no clean rx_freq/rx_time boundary for initial "
            f"{RX_FINAL_HZ/1e6:.1f} MHz tune"
        )

    if not SWEEP_ENABLE:
        return

    dwell_s = SWEEP_DWELL_MS / 1000.0
    center_hz = RX_FINAL_HZ
    loop_idx = 0
    prev_tune_ts = time.perf_counter()

    while not stop_event.is_set():
        tune_start = time.perf_counter()
        changed = tb.set_center_freq(center_hz)
        tune_call_end = time.perf_counter()

        settled_ok = tb.frame_processor.is_settled()
        if changed:
            settled_ok = tb.frame_processor.wait_until_settled(TUNE_TAG_TIMEOUT_S)
        tune_ready = time.perf_counter()

        if not settled_ok:
            log(
                f"[RX][ERROR] tune_id={tb.tune_id} center={center_hz/1e6:.1f} MHz "
                f"timed out waiting for clean tagged boundary; dwell skipped"
            )
        else:
            # Dwell starts only after stale vectors and the boundary vector have been
            # discarded. Every published frame now belongs to this center frequency.
            time.sleep(dwell_s)

        dwell_end = time.perf_counter()

        center_hz += RX_STEP_HZ
        if center_hz > RX_STOP_HZ:
            center_hz = RX_START_HZ

        loop_idx += 1
        if MEASURE_SWEEP_TIMING and (loop_idx % max(1, SWEEP_TIMING_LOG_EVERY) == 0):
            tune_call_ms = (tune_call_end - tune_start) * 1000.0
            tag_wait_ms = (tune_ready - tune_call_end) * 1000.0
            dwell_ms = (dwell_end - tune_ready) * 1000.0 if settled_ok else 0.0
            interval_ms = (tune_start - prev_tune_ts) * 1000.0
            log(
                "[TIMING] loop={} | target_dwell={:.3f} ms | actual_dwell={:.3f} ms | "
                "tune_call={:.3f} ms | tag_wait={:.3f} ms | interval_between_tunes={:.3f} ms".format(
                    loop_idx,
                    SWEEP_DWELL_MS,
                    dwell_ms,
                    tune_call_ms,
                    tag_wait_ms,
                    interval_ms,
                )
            )

        prev_tune_ts = tune_start


def _cleanup_shm(*names):
    """Best-effort removal of shared memory segments."""
    from multiprocessing.shared_memory import SharedMemory
    for name in names:
        try:
            shm = SharedMemory(name=name, create=False)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


def main():
    # ── Pin main process to cores 0-1 ────────────────────────────────────
    try:
        os.sched_setaffinity(0, CORES_MAIN)
    except OSError:
        pass

    reset_log_file()

    # ── Create shared-memory ring buffers ─────────────────────────────────
    sig_ring = ShmRingBuffer(SHM_SIG_NAME, SHM_SIG_SLOTS, SHM_SIG_SLOT_SIZE, create=True)
    fft_ring = ShmRingBuffer(SHM_FFT_NAME, SHM_FFT_SLOTS, SHM_FFT_SLOT_SIZE, create=True)
    atexit.register(_cleanup_shm, SHM_SIG_NAME, SHM_FFT_NAME)

    # ── Spawn ZMQ publisher processes ─────────────────────────────────────
    sig_pub = multiprocessing.Process(
        target=_signal_publisher_process,
        args=(SHM_SIG_NAME, SHM_SIG_SLOTS, SHM_SIG_SLOT_SIZE, ZMQ_SIGNAL_PORT, ZMQ_SIGNAL_HWM),
        daemon=True,
        name="sig_publisher",
    )
    fft_pub = multiprocessing.Process(
        target=_fft_publisher_process,
        args=(SHM_FFT_NAME, SHM_FFT_SLOTS, SHM_FFT_SLOT_SIZE, ZMQ_FFT_PORT, ZMQ_FFT_HWM),
        daemon=True,
        name="fft_publisher",
    )
    sig_pub.start()
    fft_pub.start()
    log(f"[ZMQ] signal_publisher pid={sig_pub.pid} core={CORE_SIG_PUB}")
    log(f"[ZMQ] fft_publisher pid={fft_pub.pid} core={CORE_FFT_PUB}")

    # ── GNU Radio top block (runs in main process) ────────────────────────
    tb = RxTopBlock(sig_ring=sig_ring, fft_ring=fft_ring)
    log_thread = threading.Thread(target=_log_writer_worker, daemon=True)
    telem_thread = threading.Thread(target=_telemetry_publisher_thread, args=(tb, ZMQ_TELEM_PORT), daemon=True)
    control_thread = threading.Thread(
        target=_control_listener_thread,
        args=(tb, ZMQ_CONTROL_PORT),
        daemon=True,
        name="control_listener",
    )

    def _sig_handler(sig, frame):
        stop_event.set()

    signal_mod.signal(signal_mod.SIGINT, _sig_handler)
    signal_mod.signal(signal_mod.SIGTERM, _sig_handler)

    log_thread.start()
    telem_thread.start()
    control_thread.start()
    tb.start()

    ctrl_thread = threading.Thread(target=run_warmup_and_optional_sweep, args=(tb,), daemon=True)
    ctrl_thread.start()

    while not stop_event.is_set():
        time.sleep(0.2)

    # ── Shutdown ──────────────────────────────────────────────────────────
    tb.stop()
    tb.wait()
    ctrl_thread.join(timeout=1.0)

    # Terminate publisher processes
    for p in (sig_pub, fft_pub):
        if p.is_alive():
            p.terminate()
            p.join(timeout=2.0)

    # Wait for telemetry thread to stop
    telem_thread.join(timeout=2.0)
    control_thread.join(timeout=2.0)

    log_thread.join(timeout=2.0)

    # Release shared memory
    sig_ring.unlink()
    fft_ring.unlink()


if __name__ == "__main__":
    main()
