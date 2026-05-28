#!/usr/bin/env python3
"""
ZMQ receiver for spectrum analyzer streams with HDF5 storage.

Architecture:
  - Main thread: ZMQ recv → enqueue to 2 bounded deques (never blocks on disk)
  - FFT writer thread: drains fft_queue → HDF5 direct chunk write (N=32 packing)
  - Signal writer thread: drains sig_queue → HDF5 vlen string batched write
    - Telemetry writer thread: drains telem_queue → HDF5 batched telemetry records

File rotation every 10 minutes.

Usage:
    python3 zmq_receiver.py 100.69.245.69
    python3 zmq_receiver.py 100.69.245.69 --output-dir ./data
"""

import argparse
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import zmq


# ── Configuration ─────────────────────────────────────────────────────────────

FFT_LEN = 2048
FFT_CHUNK_N = 32              # pack 32 frames into one HDF5 chunk
FILE_ROTATION_SEC = 600       # rotate files every 10 minutes
FFT_QUEUE_MAXLEN = 5000       # ~10s buffer at 488 fps
SIG_QUEUE_MAXLEN = 5000
TELEM_QUEUE_MAXLEN = 1000
SIG_BATCH_INTERVAL_S = 0.1    # flush signal batch every 100ms
FFT_FLUSH_EVERY = 1000        # flush HDF5 every 1000 rows
TELEM_BATCH_INTERVAL_S = 0.5

# ── Shared state ──────────────────────────────────────────────────────────────

stop_event = threading.Event()
fft_queue = deque(maxlen=FFT_QUEUE_MAXLEN)
sig_queue = deque(maxlen=SIG_QUEUE_MAXLEN)
telem_queue = deque(maxlen=TELEM_QUEUE_MAXLEN)

# Drop counters
fft_drops = 0
sig_drops = 0
telem_drops = 0


def dequantize_fft(fft_uint8: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    """Convert uint8 quantized FFT back to dB values."""
    return fft_uint8.astype(np.float32) / 255.0 * (db_max - db_min) + db_min


# ── HDF5 file creation helpers ────────────────────────────────────────────────

def create_fft_file(output_dir: Path) -> tuple:
    """Create a new HDF5 file for FFT data. Returns (file, datasets dict, filepath)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"fft_{ts}.h5"

    f = h5py.File(filepath, "w")

    # Main FFT dataset — chunked (32, 2048) for direct chunk write
    ds_fft = f.create_dataset(
        "fft_data",
        shape=(0, FFT_LEN),
        maxshape=(None, FFT_LEN),
        chunks=(FFT_CHUNK_N, FFT_LEN),
        dtype=np.uint8,
    )

    # Metadata arrays (one row per FFT frame)
    ds_ts = f.create_dataset(
        "timestamps", shape=(0,), maxshape=(None,), chunks=(FFT_CHUNK_N,), dtype=np.float64
    )
    ds_center = f.create_dataset(
        "center_hz", shape=(0,), maxshape=(None,), chunks=(FFT_CHUNK_N,), dtype=np.float64
    )
    ds_tune_id = f.create_dataset(
        "tune_id", shape=(0,), maxshape=(None,), chunks=(FFT_CHUNK_N,), dtype=np.uint32
    )

    # File-level attributes
    f.attrs["sample_rate"] = 50e6
    f.attrs["fft_len"] = FFT_LEN
    f.attrs["chunk_n"] = FFT_CHUNK_N
    f.attrs["created"] = datetime.now(timezone.utc).isoformat()

    return f, {"fft": ds_fft, "ts": ds_ts, "center": ds_center, "tune_id": ds_tune_id}, filepath


def create_sig_file(output_dir: Path) -> tuple:
    """Create a new HDF5 file for signal detections. Returns (file, datasets dict, filepath)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"signals_{ts}.h5"

    f = h5py.File(filepath, "w")

    # Variable-length string dataset for JSON events
    vlen_str = h5py.string_dtype()
    ds_events = f.create_dataset(
        "events", shape=(0,), maxshape=(None,), chunks=(64,), dtype=vlen_str
    )
    ds_ts = f.create_dataset(
        "timestamps", shape=(0,), maxshape=(None,), chunks=(64,), dtype=np.float64
    )
    ds_center = f.create_dataset(
        "center_hz", shape=(0,), maxshape=(None,), chunks=(64,), dtype=np.float64
    )

    f.attrs["created"] = datetime.now(timezone.utc).isoformat()

    return f, {"events": ds_events, "ts": ds_ts, "center": ds_center}, filepath


def create_telem_file(output_dir: Path) -> tuple:
    """Create a new HDF5 file for telemetry data. Returns (file, datasets dict, filepath)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"telemetry_{ts}.h5"

    f = h5py.File(filepath, "w")

    vlen_str = h5py.string_dtype()
    ds_raw = f.create_dataset(
        "events", shape=(0,), maxshape=(None,), chunks=(64,), dtype=vlen_str
    )
    ds_ts = f.create_dataset(
        "timestamps", shape=(0,), maxshape=(None,), chunks=(64,), dtype=np.float64
    )
    ds_temp = f.create_dataset(
        "temp_c", shape=(0,), maxshape=(None,), chunks=(64,), dtype=np.float64
    )
    ds_cpu = f.create_dataset(
        "cpu_percent", shape=(0,), maxshape=(None,), chunks=(64,), dtype=np.float64
    )
    ds_ram_pct = f.create_dataset(
        "ram_percent", shape=(0,), maxshape=(None,), chunks=(64,), dtype=np.float64
    )
    ds_ram_used = f.create_dataset(
        "ram_used_mb", shape=(0,), maxshape=(None,), chunks=(64,), dtype=np.float64
    )
    ds_uptime = f.create_dataset(
        "uptime_s", shape=(0,), maxshape=(None,), chunks=(64,), dtype=np.float64
    )
    ds_center = f.create_dataset(
        "center_hz", shape=(0,), maxshape=(None,), chunks=(64,), dtype=np.float64
    )
    ds_tune_id = f.create_dataset(
        "tune_id", shape=(0,), maxshape=(None,), chunks=(64,), dtype=np.uint32
    )
    ds_gain = f.create_dataset(
        "gain_db", shape=(0,), maxshape=(None,), chunks=(64,), dtype=np.float64
    )

    f.attrs["created"] = datetime.now(timezone.utc).isoformat()

    return (
        f,
        {
            "raw": ds_raw,
            "ts": ds_ts,
            "temp": ds_temp,
            "cpu": ds_cpu,
            "ram_pct": ds_ram_pct,
            "ram_used": ds_ram_used,
            "uptime": ds_uptime,
            "center": ds_center,
            "tune_id": ds_tune_id,
            "gain": ds_gain,
        },
        filepath,
    )


# ── FFT writer thread ─────────────────────────────────────────────────────────

def fft_writer_thread(output_dir: Path):
    """
    Drain fft_queue, accumulate N=32 frames, write one direct chunk.
    Rotate file every FILE_ROTATION_SEC.
    """
    f, ds, filepath = create_fft_file(output_dir)
    print(f"[FFT-WRITER] Opened {filepath.name}")

    row = 0
    file_start = time.monotonic()

    # Accumulator for packing N frames into one chunk
    acc_fft = np.zeros((FFT_CHUNK_N, FFT_LEN), dtype=np.uint8)
    acc_ts = np.zeros(FFT_CHUNK_N, dtype=np.float64)
    acc_center = np.zeros(FFT_CHUNK_N, dtype=np.float64)
    acc_tune = np.zeros(FFT_CHUNK_N, dtype=np.uint32)
    acc_idx = 0

    last_activity = time.monotonic()

    while not stop_event.is_set():
        # Drain queue
        item = None
        try:
            item = fft_queue.popleft()
        except IndexError:
            # Queue empty — check if we should flush partial accumulator
            now = time.monotonic()
            if acc_idx > 0 and (now - last_activity) > 0.2:
                # Flush partial chunk (pad with zeros — reader uses row count)
                _flush_fft_chunk(f, ds, acc_fft, acc_ts, acc_center, acc_tune, acc_idx, row)
                row += acc_idx
                acc_idx = 0
            time.sleep(0.001)
            continue

        last_activity = time.monotonic()
        meta, fft_uint8 = item

        # Store in accumulator
        acc_fft[acc_idx] = fft_uint8
        acc_ts[acc_idx] = meta.get("ts_epoch", time.time())
        acc_center[acc_idx] = meta.get("center_hz", 0)
        acc_tune[acc_idx] = meta.get("tune_id", 0)
        acc_idx += 1

        # Full chunk — write direct
        if acc_idx == FFT_CHUNK_N:
            _flush_fft_chunk(f, ds, acc_fft, acc_ts, acc_center, acc_tune, FFT_CHUNK_N, row)
            row += FFT_CHUNK_N
            acc_idx = 0

        # Periodic flush
        if row > 0 and row % FFT_FLUSH_EVERY == 0:
            f.flush()

        # File rotation
        if (time.monotonic() - file_start) >= FILE_ROTATION_SEC:
            # Flush any partial
            if acc_idx > 0:
                _flush_fft_chunk(f, ds, acc_fft, acc_ts, acc_center, acc_tune, acc_idx, row)
                row += acc_idx
                acc_idx = 0
            f.flush()
            f.close()
            print(f"[FFT-WRITER] Closed {filepath.name} ({row} frames)")
            f, ds, filepath = create_fft_file(output_dir)
            print(f"[FFT-WRITER] Opened {filepath.name}")
            row = 0
            file_start = time.monotonic()

    # Final flush on shutdown
    if acc_idx > 0:
        _flush_fft_chunk(f, ds, acc_fft, acc_ts, acc_center, acc_tune, acc_idx, row)
        row += acc_idx
    f.flush()
    f.close()
    print(f"[FFT-WRITER] Final close {filepath.name} ({row} frames)")


def _flush_fft_chunk(f, ds, acc_fft, acc_ts, acc_center, acc_tune, count, row):
    """Write accumulated frames to HDF5. Uses direct chunk write for full chunks."""
    new_size = row + count
    ds["fft"].resize(new_size, axis=0)
    ds["ts"].resize(new_size, axis=0)
    ds["center"].resize(new_size, axis=0)
    ds["tune_id"].resize(new_size, axis=0)

    if count == FFT_CHUNK_N and (row % FFT_CHUNK_N == 0):
        # Full chunk AND row is chunk-aligned — use direct chunk write
        ds["fft"].id.write_direct_chunk((row, 0), acc_fft.tobytes())
    else:
        # Partial chunk or misaligned row — use normal write
        ds["fft"][row:new_size] = acc_fft[:count]

    # Metadata always written normally (tiny, no bottleneck)
    ds["ts"][row:new_size] = acc_ts[:count]
    ds["center"][row:new_size] = acc_center[:count]
    ds["tune_id"][row:new_size] = acc_tune[:count]


# ── Signal writer thread ──────────────────────────────────────────────────────

def sig_writer_thread(output_dir: Path):
    """
    Drain sig_queue in batches, write vlen strings to HDF5.
    Rotate file every FILE_ROTATION_SEC.
    """
    f, ds, filepath = create_sig_file(output_dir)
    print(f"[SIG-WRITER] Opened {filepath.name}")

    row = 0
    file_start = time.monotonic()
    last_flush = time.monotonic()

    batch_json = []
    batch_ts = []
    batch_center = []

    while not stop_event.is_set():
        # Drain all available items from queue
        drained = 0
        while drained < 100:
            try:
                item = sig_queue.popleft()
            except IndexError:
                break
            batch_json.append(item["raw_json"])
            batch_ts.append(item.get("ts_epoch", time.time()))
            batch_center.append(item.get("center_hz", 0))
            drained += 1

        now = time.monotonic()

        # Flush batch if interval elapsed or batch large enough
        if batch_json and (now - last_flush >= SIG_BATCH_INTERVAL_S or len(batch_json) >= 100):
            new_size = row + len(batch_json)
            ds["events"].resize(new_size, axis=0)
            ds["ts"].resize(new_size, axis=0)
            ds["center"].resize(new_size, axis=0)

            ds["events"][row:new_size] = batch_json
            ds["ts"][row:new_size] = np.array(batch_ts, dtype=np.float64)
            ds["center"][row:new_size] = np.array(batch_center, dtype=np.float64)

            row += len(batch_json)
            batch_json.clear()
            batch_ts.clear()
            batch_center.clear()
            last_flush = now
            f.flush()

        # File rotation
        if (now - file_start) >= FILE_ROTATION_SEC:
            f.flush()
            f.close()
            print(f"[SIG-WRITER] Closed {filepath.name} ({row} events)")
            f, ds, filepath = create_sig_file(output_dir)
            print(f"[SIG-WRITER] Opened {filepath.name}")
            row = 0
            file_start = time.monotonic()

        if drained == 0:
            time.sleep(0.005)

    # Final flush
    if batch_json:
        new_size = row + len(batch_json)
        ds["events"].resize(new_size, axis=0)
        ds["ts"].resize(new_size, axis=0)
        ds["center"].resize(new_size, axis=0)
        ds["events"][row:new_size] = batch_json
        ds["ts"][row:new_size] = np.array(batch_ts, dtype=np.float64)
        ds["center"][row:new_size] = np.array(batch_center, dtype=np.float64)
        row += len(batch_json)
    f.flush()
    f.close()
    print(f"[SIG-WRITER] Final close {filepath.name} ({row} events)")


# ── Telemetry writer thread ───────────────────────────────────────────────────

def telem_writer_thread(output_dir: Path):
    """Drain telem_queue, write telemetry JSON + parsed fields to HDF5."""
    f, ds, filepath = create_telem_file(output_dir)
    print(f"[TELEM-WRITER] Opened {filepath.name}")

    row = 0
    file_start = time.monotonic()
    last_flush = time.monotonic()

    batch_raw = []
    batch_ts = []
    batch_temp = []
    batch_cpu = []
    batch_ram_pct = []
    batch_ram_used = []
    batch_uptime = []
    batch_center = []
    batch_tune_id = []
    batch_gain = []

    def _to_float(v):
        try:
            if v is None:
                return np.nan
            return float(v)
        except Exception:
            return np.nan

    def _to_uint32(v):
        try:
            if v is None:
                return np.uint32(0)
            return np.uint32(int(v))
        except Exception:
            return np.uint32(0)

    while not stop_event.is_set():
        drained = 0
        while drained < 100:
            try:
                item = telem_queue.popleft()
            except IndexError:
                break

            batch_raw.append(item["raw_json"])
            batch_ts.append(item.get("ts_epoch", time.time()))
            batch_temp.append(_to_float(item.get("temp_c")))
            batch_cpu.append(_to_float(item.get("cpu_percent")))
            batch_ram_pct.append(_to_float(item.get("ram_percent")))
            batch_ram_used.append(_to_float(item.get("ram_used_mb")))
            batch_uptime.append(_to_float(item.get("uptime_s")))
            batch_center.append(_to_float(item.get("center_hz")))
            batch_tune_id.append(_to_uint32(item.get("tune_id")))
            batch_gain.append(_to_float(item.get("gain_db")))
            drained += 1

        now = time.monotonic()
        if batch_raw and (now - last_flush >= TELEM_BATCH_INTERVAL_S or len(batch_raw) >= 100):
            new_size = row + len(batch_raw)
            for name in ds:
                ds[name].resize(new_size, axis=0)

            ds["raw"][row:new_size] = batch_raw
            ds["ts"][row:new_size] = np.array(batch_ts, dtype=np.float64)
            ds["temp"][row:new_size] = np.array(batch_temp, dtype=np.float64)
            ds["cpu"][row:new_size] = np.array(batch_cpu, dtype=np.float64)
            ds["ram_pct"][row:new_size] = np.array(batch_ram_pct, dtype=np.float64)
            ds["ram_used"][row:new_size] = np.array(batch_ram_used, dtype=np.float64)
            ds["uptime"][row:new_size] = np.array(batch_uptime, dtype=np.float64)
            ds["center"][row:new_size] = np.array(batch_center, dtype=np.float64)
            ds["tune_id"][row:new_size] = np.array(batch_tune_id, dtype=np.uint32)
            ds["gain"][row:new_size] = np.array(batch_gain, dtype=np.float64)

            row += len(batch_raw)
            batch_raw.clear()
            batch_ts.clear()
            batch_temp.clear()
            batch_cpu.clear()
            batch_ram_pct.clear()
            batch_ram_used.clear()
            batch_uptime.clear()
            batch_center.clear()
            batch_tune_id.clear()
            batch_gain.clear()
            last_flush = now
            f.flush()

        if (now - file_start) >= FILE_ROTATION_SEC:
            f.flush()
            f.close()
            print(f"[TELEM-WRITER] Closed {filepath.name} ({row} events)")
            f, ds, filepath = create_telem_file(output_dir)
            print(f"[TELEM-WRITER] Opened {filepath.name}")
            row = 0
            file_start = time.monotonic()

        if drained == 0:
            time.sleep(0.005)

    if batch_raw:
        new_size = row + len(batch_raw)
        for name in ds:
            ds[name].resize(new_size, axis=0)
        ds["raw"][row:new_size] = batch_raw
        ds["ts"][row:new_size] = np.array(batch_ts, dtype=np.float64)
        ds["temp"][row:new_size] = np.array(batch_temp, dtype=np.float64)
        ds["cpu"][row:new_size] = np.array(batch_cpu, dtype=np.float64)
        ds["ram_pct"][row:new_size] = np.array(batch_ram_pct, dtype=np.float64)
        ds["ram_used"][row:new_size] = np.array(batch_ram_used, dtype=np.float64)
        ds["uptime"][row:new_size] = np.array(batch_uptime, dtype=np.float64)
        ds["center"][row:new_size] = np.array(batch_center, dtype=np.float64)
        ds["tune_id"][row:new_size] = np.array(batch_tune_id, dtype=np.uint32)
        ds["gain"][row:new_size] = np.array(batch_gain, dtype=np.float64)
        row += len(batch_raw)

    f.flush()
    f.close()
    print(f"[TELEM-WRITER] Final close {filepath.name} ({row} events)")


# ── Main: ZMQ receive loop ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZMQ spectrum receiver with HDF5 storage")
    parser.add_argument("host", help="Pi5 IP address (e.g. 192.168.1.100)")
    parser.add_argument("--sig-port", type=int, default=5555, help="Signal ZMQ port")
    parser.add_argument("--fft-port", type=int, default=5556, help="FFT ZMQ port")
    parser.add_argument("--output-dir", type=str, default="./spectrum_data",
                        help="Output directory for HDF5 files")
    parser.add_argument("--telem-port", type=int, default=5557, help="Telemetry ZMQ port")
    parser.add_argument(
        "--drop-zero-frames",
        dest="drop_zero_frames",
        action="store_true",
        help="Skip FFT frames that are entirely zero (default: enabled)",
    )
    parser.add_argument(
        "--no-drop-zero-frames",
        dest="drop_zero_frames",
        action="store_false",
        help="Do not skip zero FFT frames",
    )
    parser.set_defaults(drop_zero_frames=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start writer threads
    t_fft = threading.Thread(target=fft_writer_thread, args=(output_dir,), daemon=True, name="fft_writer")
    t_sig = threading.Thread(target=sig_writer_thread, args=(output_dir,), daemon=True, name="sig_writer")
    t_telem = threading.Thread(target=telem_writer_thread, args=(output_dir,), daemon=True, name="telem_writer")
    t_fft.start()
    t_sig.start()
    t_telem.start()

    # ZMQ setup
    ctx = zmq.Context()

    sig_sock = ctx.socket(zmq.SUB)
    sig_sock.setsockopt(zmq.SUBSCRIBE, b"")
    sig_sock.setsockopt(zmq.RCVHWM, 500)
    sig_sock.connect(f"tcp://{args.host}:{args.sig_port}")

    telem_sock = ctx.socket(zmq.SUB)
    telem_sock.setsockopt(zmq.SUBSCRIBE, b"")
    telem_sock.setsockopt(zmq.RCVHWM, 500)
    telem_sock.connect(f"tcp://{args.host}:{args.telem_port}")

    fft_sock = ctx.socket(zmq.SUB)
    fft_sock.setsockopt(zmq.SUBSCRIBE, b"")
    fft_sock.setsockopt(zmq.RCVHWM, 200)
    fft_sock.connect(f"tcp://{args.host}:{args.fft_port}")

    poller = zmq.Poller()
    poller.register(sig_sock, zmq.POLLIN)
    poller.register(fft_sock, zmq.POLLIN)
    poller.register(telem_sock, zmq.POLLIN)

    global fft_drops, sig_drops
    sig_count = 0
    fft_count = 0
    telem_count = 0
    t0 = time.monotonic()

    print(f"[RECV] Connected to {args.host}  sig:{args.sig_port}  fft:{args.fft_port}  telem:{args.telem_port}")
    print(f"[RECV] Storing to {output_dir.resolve()}")
    print(f"[RECV] drop_zero_frames={args.drop_zero_frames}")
    print("-" * 72)

    try:
        while True:
            events = dict(poller.poll(timeout=100))

            if sig_sock in events:
                raw = sig_sock.recv()
                try:
                    sig = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                sig_count += 1
                # Parse timestamp to epoch float
                ts_str = sig.get("ts", "")
                try:
                    ts_epoch = datetime.fromisoformat(ts_str).timestamp()
                except (ValueError, OSError):
                    ts_epoch = time.time()

                # Enqueue for writer
                sig_queue.append({
                    "raw_json": raw.decode("utf-8"),
                    "ts_epoch": ts_epoch,
                    "center_hz": sig.get("center_hz", 0),
                })

                # Console output
                n = len(sig.get("signals", []))
                center_mhz = sig.get("center_hz", 0) / 1e6
                if sig_count % 10 == 1:
                    print(
                        f"[SIG] #{sig_count}  center={center_mhz:.1f} MHz  "
                        f"signals={n}  noise={sig.get('noise_floor_db', 0):+.1f} dB"
                    )

            if telem_sock in events:
                raw = telem_sock.recv()
                try:
                    telem = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                telem_count += 1
                ts_str = telem.get("ts", "")
                try:
                    ts_epoch = datetime.fromisoformat(ts_str).timestamp()
                except (ValueError, OSError):
                    ts_epoch = time.time()

                app_state = telem.get("app", {}) or {}
                telem_queue.append({
                    "raw_json": raw.decode("utf-8"),
                    "ts_epoch": ts_epoch,
                    "temp_c": telem.get("temp_c"),
                    "cpu_percent": telem.get("cpu_percent"),
                    "ram_percent": telem.get("ram_percent"),
                    "ram_used_mb": telem.get("ram_used_mb"),
                    "uptime_s": telem.get("uptime_s"),
                    "center_hz": app_state.get("center_hz", 0),
                    "tune_id": app_state.get("tune_id", 0),
                    "gain_db": app_state.get("gain_db", 0),
                })

                if telem_count % 10 == 1:
                    temp_c = telem.get("temp_c", None)
                    cpu_pct = telem.get("cpu_percent", None)
                    ram_pct = telem.get("ram_percent", None)
                    gain_db = app_state.get("gain_db", None)
                    print(
                        f"[TELEM] #{telem_count} temp={temp_c} C cpu={cpu_pct}% ram={ram_pct}% gain={gain_db} dB"
                    )

            if fft_sock in events:
                parts = fft_sock.recv_multipart()
                if len(parts) != 2:
                    continue

                try:
                    meta = json.loads(parts[0])
                except json.JSONDecodeError:
                    continue
                fft_uint8 = np.frombuffer(parts[1], dtype=np.uint8)
                if len(fft_uint8) != FFT_LEN:
                    continue

                # Optionally drop frames that are entirely zero to avoid storing blank data
                if args.drop_zero_frames:
                    try:
                        if np.all(fft_uint8 == 0):
                            fft_drops += 1
                            continue
                    except Exception:
                        # In case of any unexpected error in the quick check, fall back to keeping the frame
                        pass

                fft_count += 1

                # Parse timestamp
                ts_str = meta.get("ts", "")
                try:
                    ts_epoch = datetime.fromisoformat(ts_str).timestamp()
                except (ValueError, OSError):
                    ts_epoch = time.time()

                # Enqueue for writer — store metadata + raw uint8
                fft_queue.append((
                    {
                        "ts_epoch": ts_epoch,
                        "center_hz": meta.get("center_hz", 0),
                        "tune_id": meta.get("tune_id", 0),
                        "db_min": meta.get("db_min", -120.0),
                        "db_max": meta.get("db_max", 0.0),
                    },
                    fft_uint8,
                ))

                # Periodic console status
                if fft_count % 200 == 0:
                    elapsed = time.monotonic() - t0
                    fps = fft_count / elapsed if elapsed > 0 else 0
                    q_fft = len(fft_queue)
                    q_sig = len(sig_queue)
                    print(
                        f"[FFT] #{fft_count}  {fps:.1f} fps  "
                        f"queues: fft={q_fft}/{FFT_QUEUE_MAXLEN} sig={q_sig}/{SIG_QUEUE_MAXLEN} "
                        f"drops={fft_drops}"
                    )

    except KeyboardInterrupt:
        elapsed = time.monotonic() - t0
        print(f"\n[RECV] Stopping... {sig_count} signals, {fft_count} FFT frames in {elapsed:.1f}s")
    finally:
        stop_event.set()
        sig_sock.close()
        fft_sock.close()
        ctx.term()

        # Wait for writers to flush
        t_fft.join(timeout=5.0)
        t_sig.join(timeout=5.0)
        t_telem.join(timeout=5.0)
        print("[RECV] Writers shut down. Done.")


if __name__ == "__main__":
    main()
