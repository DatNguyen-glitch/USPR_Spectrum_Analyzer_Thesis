#!/usr/bin/env python3
"""Terminal UI for telemetry monitoring and gain control.

- SUB telemetry from tcp://<host>:5557
- PUSH control commands to tcp://<host>:5558
"""

import argparse
import json
import re
import threading
import time
from dataclasses import dataclass, field

import curses
import zmq


GAIN_CMD_RE = re.compile(r"^gain:\s*([+-]?(?:\d+\.?\d*|\d*\.\d+))$")


@dataclass
class TelemetryState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    data: dict = field(default_factory=dict)
    updated_ts: float = 0.0


def telemetry_worker(ctx: zmq.Context, host: str, port: int, state: TelemetryState, stop_event: threading.Event):
    """Receive telemetry JSON and update shared state."""
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVHWM, 500)
    sock.connect(f"tcp://{host}:{port}")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    while not stop_event.is_set():
        events = dict(poller.poll(timeout=200))
        if sock in events:
            raw = sock.recv()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            with state.lock:
                state.data = payload
                state.updated_ts = time.time()

    sock.close()


def _fmt_float(value, digits=1, suffix=""):
    try:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return "n/a"


def _fmt_int(value):
    try:
        if value is None:
            return "n/a"
        return f"{int(value)}"
    except Exception:
        return "n/a"


def tui_main(stdscr, host: str, telem_port: int, control_port: int):
    """Run curses-based TUI loop."""
    stop_event = threading.Event()
    ctx = zmq.Context()
    state = TelemetryState()

    # Control socket (PUSH) for one-way commands
    control_sock = ctx.socket(zmq.PUSH)
    control_sock.setsockopt(zmq.SNDHWM, 10)
    control_sock.connect(f"tcp://{host}:{control_port}")

    recv_thread = threading.Thread(
        target=telemetry_worker,
        args=(ctx, host, telem_port, state, stop_event),
        daemon=True,
        name="telemetry_worker",
    )
    recv_thread.start()

    curses.curs_set(1)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    input_buf = ""
    error_msg = ""
    info_msg = ""

    try:
        while True:
            ch = stdscr.getch()
            if ch != -1:
                if ch in (10, 13):
                    cmd = input_buf.strip()
                    input_buf = ""
                    if not cmd:
                        continue
                    match = GAIN_CMD_RE.match(cmd)
                    if not match:
                        error_msg = "Error: Only 'gain: <value>' command is allowed."
                        info_msg = ""
                    else:
                        gain_value = float(match.group(1))
                        payload = {"action": "set_gain", "value": gain_value}
                        try:
                            control_sock.send_json(payload, flags=zmq.NOBLOCK)
                            info_msg = f"Sent set_gain {gain_value:.2f}"
                            error_msg = ""
                        except zmq.Again:
                            error_msg = "Error: Control socket busy. Try again."
                            info_msg = ""
                elif ch in (curses.KEY_BACKSPACE, 127, 8):
                    input_buf = input_buf[:-1]
                elif ch == 3:  # Ctrl+C
                    raise KeyboardInterrupt
                elif 32 <= ch <= 126:
                    input_buf += chr(ch)

            # Snapshot telemetry for rendering
            with state.lock:
                data = dict(state.data) if state.data else {}
                updated_ts = state.updated_ts

            ts_str = data.get("ts", "n/a")
            cpu = _fmt_float(data.get("cpu_percent"), digits=1, suffix="%")
            ram_pct = _fmt_float(data.get("ram_percent"), digits=1, suffix="%")
            ram_used = _fmt_float(data.get("ram_used_mb"), digits=1, suffix=" MB")
            temp_c = _fmt_float(data.get("temp_c"), digits=1, suffix=" C")
            uptime = _fmt_float(data.get("uptime_s"), digits=1, suffix=" s")

            app = data.get("app", {}) or {}
            center_hz = _fmt_float(app.get("center_hz"), digits=1, suffix=" Hz")
            tune_id = _fmt_int(app.get("tune_id"))
            gain_db = _fmt_float(app.get("gain_db"), digits=1, suffix=" dB")

            now = time.time()
            age_s = "n/a"
            if updated_ts:
                age_s = _fmt_float(max(0.0, now - updated_ts), digits=1, suffix=" s")

            stdscr.erase()
            h, w = stdscr.getmaxyx()

            lines = [
                "Telemetry UI (Ctrl+C to exit)",
                f"Last update age: {age_s}",
                f"Timestamp: {ts_str}",
                f"CPU: {cpu} | RAM: {ram_pct} ({ram_used}) | Temp: {temp_c} | Uptime: {uptime}",
                f"Center: {center_hz} | Tune ID: {tune_id} | Gain: {gain_db}",
                "-" * max(0, w - 1),
                f"Info: {info_msg}" if info_msg else "Info:",
                f"Error: {error_msg}" if error_msg else "Error:",
                f"Command> {input_buf}",
            ]

            for idx, line in enumerate(lines[: h - 1]):
                stdscr.addnstr(idx, 0, line, w - 1)

            stdscr.refresh()

    finally:
        stop_event.set()
        recv_thread.join(timeout=1.0)
        control_sock.close()
        ctx.term()


def main():
    parser = argparse.ArgumentParser(description="Telemetry TUI for USRP sweep")
    parser.add_argument("host", help="Pi5 IP address (e.g. 192.168.1.100)")
    parser.add_argument("--telem-port", type=int, default=5557, help="Telemetry ZMQ port")
    parser.add_argument("--control-port", type=int, default=5558, help="Control ZMQ port")
    args = parser.parse_args()

    curses.wrapper(tui_main, args.host, args.telem_port, args.control_port)


if __name__ == "__main__":
    main()
