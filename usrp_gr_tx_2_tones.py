#!/usr/bin/env python3
"""
TX-only controller using GNU Radio top_block (no GUI).
"""

import signal
import threading
import time
import select
import sys
from datetime import datetime

import pmt
# THÊM THƯ VIỆN analog ĐỂ TẠO SÓNG SIN
from gnuradio import blocks, gr, uhd, analog 

TX_SERIAL = "34D62A7"

# ==============================================================================
# --- OLD QPSK CONFIGURATION (COMMENTED OUT) ---
# TX_FILE = "/home/datnguyen/Desktop/data_0406.complex_float"
# TX_SAMP_RATE = 2e6
# TX_GAIN_DB = 60
# TX_BW_HZ = 4e6

# ==============================================================================
# --- NEW SINE WAVE (CW) CONFIGURATION ---
TX_SAMP_RATE = 8e6          # Lấy mẫu 5M là quá đủ cho sóng đơn tần
TX_GAIN_DB = 15             # CRITICAL: Bắt buộc hạ Gain xuống 35-40 để chống bão hòa ADC bên máy thu
TX_BW_HZ = 8e6              # Mở bộ lọc analog 5M
IF_OFFSET_HZ = 1.5e6          # Dịch tần baseband 1MHz để né nhiễu LO Leakage
TX_AMPLITUDE = 0.15          # Biên độ số 0.3 để tránh tràn DAC
TONE_SPACING_HZ = 6e4     # KHOẢNG CÁCH 2 ĐỈNH SIN: Ví dụ 100 kHz (Mỗi đỉnh cách tâm 50 kHz)
# ==============================================================================

TX_CENTER_HZ = 4.15e8

stop_event = threading.Event()

# Frequency shift configuration
SHIFT_STEP_HZ = 7.5e6
SHIFT_MIN_HZ = 3.585e8
SHIFT_MAX_HZ = 5.05e8
SHIFT_COOLDOWN_S = 3 * 60


class ShiftController:
    """Controls periodic and manual frequency shifts for a USRP sink."""

    def __init__(self, usrp_sink, initial_freq_hz):
        self.usrp_sink = usrp_sink
        self.current_freq = float(initial_freq_hz)
        self.lock = threading.Lock()
        self.thread = None
        self._stop = threading.Event()

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def _do_shift(self):
        with self.lock:
            width = SHIFT_MAX_HZ - SHIFT_MIN_HZ
            offset = (self.current_freq - SHIFT_MIN_HZ + SHIFT_STEP_HZ) % width
            new_freq = SHIFT_MIN_HZ + offset
            self.current_freq = new_freq
            try:
                # TRỪ ĐI IF_OFFSET_HZ ĐỂ LO LÙI LẠI, ÉP SÓNG SIN RƠI ĐÚNG VÀO CURRENT_FREQ
                lo_freq = self.current_freq - IF_OFFSET_HZ
                self.usrp_sink.set_center_freq(lo_freq, 0)
            except Exception:
                print(f"[shift] set_center_freq failed for {self.current_freq}")
            
            print(f"[{datetime.utcnow().isoformat()}] Shifted Target Freq -> {self.current_freq/1e6:.6f} MHz (Hardware LO tuned to {lo_freq/1e6:.6f} MHz)")

    def _run(self):
        next_allowed = time.time()
        while not self._stop.is_set() and not stop_event.is_set():
            now = time.time()
            timeout = max(0.0, next_allowed - now)

            try:
                rlist, _, _ = select.select([sys.stdin], [], [], timeout if timeout > 0 else 0.1)
            except Exception:
                rlist = []

            if rlist:
                try:
                    line = sys.stdin.readline()
                except Exception:
                    line = ""
                if line is None:
                    line = ""
                cmd = line.strip().lower()
                if cmd == "shift":
                    self._do_shift()
                    next_allowed = time.time() + SHIFT_COOLDOWN_S
                continue

            if time.time() >= next_allowed:
                self._do_shift()
                next_allowed = time.time() + SHIFT_COOLDOWN_S


class TxTopBlock(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "dual_usrp_sweep_gr_tx", catch_exceptions=True)

        # --- NEW TWO-TONE SINE WAVE SOURCES ---
        # Tần số 1: Dịch lùi so với IF_OFFSET một nửa khoảng cách
        freq_tone_1 = IF_OFFSET_HZ - (TONE_SPACING_HZ / 2.0)
        
        # Tần số 2: Dịch tiến so với IF_OFFSET một nửa khoảng cách
        freq_tone_2 = IF_OFFSET_HZ + (TONE_SPACING_HZ / 2.0)

        # Khởi tạo Sóng sin 1 (Complex Exponential)
        self.tx_sig_source_1 = analog.sig_source_c(
            TX_SAMP_RATE,
            analog.GR_COS_WAVE, 
            freq_tone_1,
            TX_AMPLITUDE / 2.0,  # CỰC KỲ QUAN TRỌNG: Chia đôi biên độ để tổng 2 sóng không làm tràn DAC
            0
        )

        # Khởi tạo Sóng sin 2 (Complex Exponential)
        self.tx_sig_source_2 = analog.sig_source_c(
            TX_SAMP_RATE,
            analog.GR_COS_WAVE, 
            freq_tone_2,
            TX_AMPLITUDE / 2.0,  
            0
        )

        # Khởi tạo bộ cộng luồng IQ phức (Complex Adder)
        self.adder = blocks.add_cc()

        # Khởi tạo USRP Sink
        self.usrp_sink = uhd.usrp_sink(
            ",".join((f"serial={TX_SERIAL}", f"otw_format=sc16")),
            uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc16",
                args="",
                channels=list(range(0, 1)),
            ),
            "",
        )
        self.usrp_sink.set_samp_rate(TX_SAMP_RATE)
        
        # Thiết lập LO cứng của phần cứng
        self.usrp_sink.set_center_freq(TX_CENTER_HZ - IF_OFFSET_HZ, 0)
        self.usrp_sink.set_gain(TX_GAIN_DB, 0)
        self.usrp_sink.set_antenna("TX/RX", 0)
        self.usrp_sink.set_bandwidth(TX_BW_HZ, 0)
        
        # --- KẾT NỐI LUỒNG TÍN HIỆU (FLOWGRAPH CONNECTION) ---
        # 1. Đấu sóng sin 1 vào cổng 0 của khối cộng
        self.connect((self.tx_sig_source_1, 0), (self.adder, 0))
        # 2. Đấu sóng sin 2 vào cổng 1 của khối cộng
        self.connect((self.tx_sig_source_2, 0), (self.adder, 1))
        # 3. Đấu tổng 2 sóng từ khối cộng xuống thiết bị USRP
        self.connect((self.adder, 0), (self.usrp_sink, 0))


def main():
    tb = TxTopBlock()
    shift_ctrl = ShiftController(tb.usrp_sink, TX_CENTER_HZ)

    def _sig_handler(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    tb.start()
    try:
        shift_ctrl.start()
    except Exception:
        print("Failed to start ShiftController")

    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        try:
            shift_ctrl.stop()
        except Exception:
            pass
        tb.stop()
        tb.wait()


if __name__ == "__main__":
    main()
