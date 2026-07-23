#!/usr/bin/env python3
"""
RX-only controller using GNU Radio top_block (no GUI).

Fixed-frequency variant: the frequency sweep has been removed. The receiver
tunes once to a user-supplied center frequency and can be re-tuned live by
typing a new frequency into the terminal.

Flow (unchanged from the sweep version):
- RX: uhd.usrp_source -> blocks.stream_to_vector(FFT_LEN) -> FrameProcessor
- FrameProcessor writes detected signals + quantized FFT into two shared-memory
  SPSC ring buffers.
- Two separate processes (signal_publisher, fft_publisher) read from the ring
  buffers and publish via ZMQ PUB sockets.
- Main process (GNU Radio + detection) pinned to cores 0-1.
- Signal publisher pinned to core 3, FFT publisher pinned to core 2.

Tuning:
- Initial center frequency is taken from the --freq command-line argument (MHz).
- While running, type a frequency in MHz at the terminal prompt to retune live;
  type 'q' (or 'quit'/'exit') to stop. Use --no-interactive to disable this.
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
import queue
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
from gnuradio import blocks, gr, uhd

from shm_ring_buffer import ShmRingBuffer
import psutil
import zmq


RX_SERIAL = "34D62A7"
FPGA_BIN = "/home/datnguyen/Downloads/USRP_B210_FPGA_Thesis/FPGA_bin_files/b200_copilot_14h45_2105.bin"

RX_SAMP_RATE    = 50e6
RX_BW           = 50e6
RX_GAIN_DB = 22
RX_INIT_HZ = 4e8           # USRP source init frequency (before warm-up)
RX_WARMUP_HZ = 5e8
RX_WARMUP_DELAY_S = 1
RX_WARMUP_HOLD_S = 1
RX_DEFAULT_HZ = 5e8        # default target frequency if --freq not given
FFT_LEN = 2048

# Settle time after a manual tune before re-enabling collection.
TUNE_SETTLE_S = 0.2

DETECT_MARGIN_DB = 4.0
MIN_SIGNAL_BINS = 4
MIN_PEAK_SNR_DB = 10
EDGE_GUARD_BINS = 40
DC_GUARD_BINS = 48
DC_GUARD_HZ = 1.5e6

DETECT_EVERY = 5
LOG_FILE = Path(__file__).resolve().with_name("fixed_freq_gr.log")
DEBUG_FRAME_DIR = Path(__file__).resolve().with_name("debug_frames")
DEBUG_FRAME_INDEX = 5
LOG_QUEUE_MAXSIZE = 10000
LOG_BATCH_SIZE = 200
LOG_FLUSH_INTERVAL_S = 0.5

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
SHM_FFT_SLOT_SIZE = 2048 + 512  # 2048 uint8 bins + up to 512 bytes JSON metadata

# FFT quantization
FFT_QUANTIZE_DB_MIN = -120.0
FFT_QUANTIZE_DB_MAX = 0.0

# Send every Nth FFT frame to ZMQ (bandwidth reduction)
FFT_STREAM_DECIMATION = 5

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
    """Consume vectorized complex FFT frames and run signal detection + FFT streaming."""

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
        self._discard_all = False   # when True, discard every frame
        self._settled = threading.Event()  # signals tuner that settle is done
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._detect_busy = False
        self._sig_ring = sig_ring
        self._fft_ring = fft_ring

    def begin_tune(self, tune_id: int, freq_hz: float):
        with self._cv:
            while self._detect_busy:
                self._cv.wait(timeout=0.01)
            self.tune_id = int(tune_id)
            self.center_hz = float(freq_hz)
            self.frame_epoch = 0
            self._discard_all = True
            self._settled.clear()

    def mark_settled(self):
        """Called after the dwell/settle sleep to start collecting."""
        with self._lock:
            self._discard_all = False
        self._settled.set()

    def work(self, input_items, output_items):
        frames = input_items[0]
        for i in range(len(frames)):
            vec = np.asarray(frames[i], dtype=np.complex64)
            self.frame_counter += 1
            should_detect = False

            with self._lock:
                tune_id = self.tune_id
                center_hz = self.center_hz
                if self._discard_all:
                    continue
                frame_epoch = self.frame_epoch
                self.frame_epoch += 1

                if frame_epoch + 1 == DEBUG_FRAME_INDEX:
                    save_debug_frame(vec, tune_id, center_hz, frame_epoch)

                if frame_epoch % self.detect_every == 0:
                    self._detect_busy = True
                    should_detect = True

            # ── Stream quantized FFT to ring buffer ──────────────────
            if self._fft_ring is not None and frame_epoch % FFT_STREAM_DECIMATION == 0:
                fft_uint8 = quantize_fft_to_uint8(vec)
                meta = json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "tune_id": tune_id,
                    "center_hz": center_hz,
                    "sr": RX_SAMP_RATE,
                    "fft_len": self.fft_len,
                    "db_min": FFT_QUANTIZE_DB_MIN,
                    "db_max": FFT_QUANTIZE_DB_MAX,
                    "epoch": frame_epoch,
                }, separators=(",", ":")).encode("utf-8")
                # Pack: [2-byte meta_len][meta][fft_uint8]
                slot_data = struct.pack("<H", len(meta)) + meta + fft_uint8.tobytes()
                self._fft_ring.write(slot_data)

            if not should_detect:
                continue

            try:
                noise_floor, threshold, signals, g_bin, g_freq = detect_signals(vec, center_hz)

                with self._cv:
                    # Handshake invariant: retune waits until this detect section exits.
                    if not self._discard_all and self.tune_id == tune_id and signals:
                        # ── Stream detected signals to ring buffer ────
                        if self._sig_ring is not None:
                            sig_payload = json.dumps({
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "tune_id": tune_id,
                                "center_hz": center_hz,
                                "noise_floor_db": noise_floor,
                                "threshold_db": threshold,
                                "signals": signals,
                            }, separators=(",", ":")).encode("utf-8")
                            self._sig_ring.write(sig_payload)

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
            finally:
                with self._cv:
                    self._detect_busy = False
                    self._cv.notify_all()

        return len(frames)


class RxTopBlock(gr.top_block):
    def __init__(self, sig_ring=None, fft_ring=None):
        gr.top_block.__init__(self, "usrp_fixed_freq_gr_rx", catch_exceptions=True)

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

        self.stream_to_vector = blocks.stream_to_vector(gr.sizeof_gr_complex * 1, FFT_LEN)
        self.frame_processor = FrameProcessor(
            FFT_LEN, self.center_hz, DETECT_EVERY,
            sig_ring=sig_ring, fft_ring=fft_ring,
        )

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
        # Tell FrameProcessor to discard ALL frames from now on
        self.frame_processor.begin_tune(self.tune_id, self.center_hz)
        # Hardware tune (3-107 ms) — FrameProcessor discards during this
        t0 = time.perf_counter()
        self.usrp_source.set_center_freq(self.center_hz, 0)
        t1 = time.perf_counter()
        tune_ms = (t1 - t0) * 1000
        print(f"Tune ms: {tune_ms:.1f}", flush=True)
        log(
            f"[RX] tune_id={self.tune_id} -> {self.center_hz/1e6:.1f} MHz"
        )

    def set_gain(self, gain_db: float):
        try:
            self.usrp_source.set_gain(float(gain_db), 0)
            self.current_gain = float(gain_db)
            log(f"[RX] set_gain={self.current_gain} dB")
        except Exception as e:
            log(f"[RX] set_gain_failed: {e}")


def tune_to(tb: RxTopBlock, freq_hz: float):
    """Retune the receiver and re-enable collection after a short settle.

    Mirrors the flush/collect handshake the sweep loop used: begin_tune
    (inside set_center_freq) discards stale GR-buffered frames, then after
    a settle delay mark_settled() re-enables detection on clean data.
    """
    tb.set_center_freq(freq_hz)
    time.sleep(TUNE_SETTLE_S)
    tb.frame_processor.mark_settled()


def run_warmup_and_tune(tb: RxTopBlock, target_hz: float):
    """Warm up the USRP PLL/AGC, then settle at the requested fixed frequency.

    Keeps the original warm-up sequence (init -> warmup -> target) but does
    NOT sweep: it tunes once to `target_hz` and enables collection.
    """
    time.sleep(RX_WARMUP_DELAY_S)
    if stop_event.is_set():
        return

    tb.set_center_freq(RX_WARMUP_HZ)
    time.sleep(RX_WARMUP_HOLD_S)
    if stop_event.is_set():
        return

    tb.set_center_freq(target_hz)
    tb.frame_processor.mark_settled()
    log(f"[RX] fixed_freq={target_hz/1e6:.3f} MHz (collection enabled)")


def _terminal_tune_thread(tb: RxTopBlock, target_hz: float, interactive: bool):
    """Warm up, settle at target_hz, then (optionally) read MHz from stdin to retune."""
    run_warmup_and_tune(tb, target_hz)

    if not interactive:
        return

    print(
        "\n[tuner] Enter a center frequency in MHz to retune "
        "(e.g. 868), or 'q' to quit.\n",
        flush=True,
    )
    while not stop_event.is_set():
        try:
            line = input("freq (MHz) > ").strip()
        except (EOFError, OSError):
            break  # stdin closed (e.g. running detached) — keep collecting

        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            stop_event.set()
            break

        try:
            freq_mhz = float(line)
        except ValueError:
            print("  invalid input — enter a number in MHz (e.g. 868)", flush=True)
            continue

        if freq_mhz <= 0:
            print("  frequency must be positive", flush=True)
            continue

        tune_to(tb, freq_mhz * 1e6)
        print(f"  tuned to {freq_mhz:.3f} MHz", flush=True)


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fixed-frequency USRP RX controller (no sweep). "
                    "Tune from the terminal."
    )
    parser.add_argument(
        "--freq", type=float, default=RX_DEFAULT_HZ / 1e6,
        help="Initial center frequency in MHz (default: %(default)s).",
    )
    parser.add_argument(
        "--gain", type=float, default=RX_GAIN_DB,
        help="RX gain in dB (default: %(default)s).",
    )
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="Do not read frequencies from stdin; just hold the initial frequency.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    target_hz = args.freq * 1e6

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
    tb.set_gain(args.gain)

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

    # Warm-up + terminal tuning (replaces the sweep control thread)
    tuner_thread = threading.Thread(
        target=_terminal_tune_thread,
        args=(tb, target_hz, not args.no_interactive),
        daemon=True,
        name="terminal_tuner",
    )
    tuner_thread.start()

    while not stop_event.is_set():
        time.sleep(0.2)

    # ── Shutdown ──────────────────────────────────────────────────────────
    tb.stop()
    tb.wait()
    tuner_thread.join(timeout=1.0)

    # Terminate publisher processes
    for p in (sig_pub, fft_pub):
        if p.is_alive():
            p.terminate()
            p.join(timeout=2.0)

    # Wait for telemetry / control threads to stop
    telem_thread.join(timeout=2.0)
    control_thread.join(timeout=2.0)

    log_thread.join(timeout=2.0)

    # Release shared memory
    sig_ring.unlink()
    fft_ring.unlink()


if __name__ == "__main__":
    main()