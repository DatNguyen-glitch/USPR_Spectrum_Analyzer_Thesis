#!/usr/bin/env python3
"""
TX-only controller using GNU Radio top_block (no GUI).

Flow:
- TX: blocks.file_source(complex float) -> uhd.usrp_sink
"""

import signal
import threading
import time

import pmt
from gnuradio import blocks, gr, uhd


TX_SERIAL = "34D62A7"
TX_FILE = "/home/datnguyen/Desktop/data.complex_float"

TX_SAMP_RATE = 2e6
TX_CENTER_HZ = 5.1e8
TX_GAIN_DB = 22
TX_BW_HZ = 4e6


stop_event = threading.Event()


class TxTopBlock(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "dual_usrp_sweep_gr_tx", catch_exceptions=True)

        self.tx_file_source = blocks.file_source(
            gr.sizeof_gr_complex * 1,
            TX_FILE,
            True,
            0,
            0,
        )
        self.tx_file_source.set_begin_tag(pmt.PMT_NIL)

        self.usrp_sink = uhd.usrp_sink(
            ",".join((f"serial={TX_SERIAL}", "serial=34D62A7,otw_format=sc16")),
            uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc16",
                args="",
                channels=list(range(0, 1)),
            ),
            "",
        )
        self.usrp_sink.set_samp_rate(TX_SAMP_RATE)
        self.usrp_sink.set_center_freq(TX_CENTER_HZ, 0)
        self.usrp_sink.set_gain(TX_GAIN_DB, 0)
        self.usrp_sink.set_antenna("TX/RX", 0)
        self.usrp_sink.set_bandwidth(TX_BW_HZ, 0)

        self.connect((self.tx_file_source, 0), (self.usrp_sink, 0))


def main():
    tb = TxTopBlock()

    def _sig_handler(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    tb.start()

    while not stop_event.is_set():
        time.sleep(0.2)

    tb.stop()
    tb.wait()


if __name__ == "__main__":
    main()