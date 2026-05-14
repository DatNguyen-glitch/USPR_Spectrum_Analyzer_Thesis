#!/usr/bin/env python3
"""
ZMQ receiver for spectrum analyzer streams.

Subscribes to two ZMQ PUB sockets:
  - Port 5555: Detected signals (JSON)
  - Port 5556: Quantized FFT data (multipart: [metadata_json, uint8_binary])

Run on the Host PC. Pass the Pi5 IP as the first argument:
    python3 zmq_receiver.py 192.168.1.100
"""

import argparse
import json
import sys
import time

import numpy as np
import zmq


def dequantize_fft(fft_uint8: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    """Convert uint8 quantized FFT back to dB values."""
    return fft_uint8.astype(np.float32) / 255.0 * (db_max - db_min) + db_min


def main():
    parser = argparse.ArgumentParser(description="ZMQ spectrum receiver")
    parser.add_argument("host", help="Pi5 IP address (e.g. 192.168.1.100)")
    parser.add_argument("--sig-port", type=int, default=5555, help="Signal ZMQ port")
    parser.add_argument("--fft-port", type=int, default=5556, help="FFT ZMQ port")
    args = parser.parse_args()

    ctx = zmq.Context()

    # Signal subscriber
    sig_sock = ctx.socket(zmq.SUB)
    sig_sock.setsockopt(zmq.SUBSCRIBE, b"")
    sig_sock.setsockopt(zmq.RCVHWM, 500)
    sig_sock.connect(f"tcp://{args.host}:{args.sig_port}")

    # FFT subscriber
    fft_sock = ctx.socket(zmq.SUB)
    fft_sock.setsockopt(zmq.SUBSCRIBE, b"")
    fft_sock.setsockopt(zmq.RCVHWM, 200)
    fft_sock.connect(f"tcp://{args.host}:{args.fft_port}")

    poller = zmq.Poller()
    poller.register(sig_sock, zmq.POLLIN)
    poller.register(fft_sock, zmq.POLLIN)

    sig_count = 0
    fft_count = 0
    t0 = time.monotonic()

    print(f"Listening on {args.host}  sig:{args.sig_port}  fft:{args.fft_port}")
    print("-" * 72)

    try:
        while True:
            events = dict(poller.poll(timeout=100))  # 100ms timeout

            if sig_sock in events:
                raw = sig_sock.recv()
                try:
                    sig = json.loads(raw)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Truncated signal JSON ({len(raw)} bytes): {e}")
                    continue
                sig_count += 1
                n = len(sig.get("signals", []))
                center_mhz = sig.get("center_hz", 0) / 1e6
                print(
                    f"[SIG] #{sig_count}  center={center_mhz:.1f} MHz  "
                    f"signals={n}  noise={sig.get('noise_floor_db', 0):+.1f} dB"
                )
                for s in sig.get("signals", []):
                    print(
                        f"       peak={s['peak_freq']/1e6:.3f} MHz  "
                        f"power={s['peak_db']:+.1f} dB  "
                        f"snr={s['peak_snr_db']:.1f} dB  "
                        f"bw={s['width_hz']/1e3:.1f} kHz"
                    )

            if fft_sock in events:
                parts = fft_sock.recv_multipart()
                if len(parts) == 2:
                    meta = json.loads(parts[0])
                    fft_uint8 = np.frombuffer(parts[1], dtype=np.uint8)
                    fft_count += 1

                    fft_db = dequantize_fft(
                        fft_uint8, meta.get("db_min", -120), meta.get("db_max", 0)
                    )
                    center_mhz = meta.get("center_hz", 0) / 1e6
                    peak_db = float(np.max(fft_db))

                    if fft_count % 50 == 0:
                        elapsed = time.monotonic() - t0
                        fps = fft_count / elapsed if elapsed > 0 else 0
                        print(
                            f"[FFT] #{fft_count}  center={center_mhz:.1f} MHz  "
                            f"bins={len(fft_uint8)}  peak={peak_db:+.1f} dB  "
                            f"rate={fps:.1f} fps"
                        )

    except KeyboardInterrupt:
        elapsed = time.monotonic() - t0
        print(f"\nStopped. Received {sig_count} signals, {fft_count} FFT frames in {elapsed:.1f}s")
    finally:
        sig_sock.close()
        fft_sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
