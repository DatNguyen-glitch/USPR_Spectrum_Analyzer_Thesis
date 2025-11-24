from gnuradio import gr
import numpy as np
from collections import deque
import time
import os

class ring_buffer(gr.sync_block):
    def _delete_if_empty(self, filename):
        """Delete the file if it is empty (0 bytes)."""
        try:
            if os.path.isfile(filename) and os.path.getsize(filename) == 0:
                os.remove(filename)
        except Exception:
            pass

    def __init__(self, samp_rate, buffer_ms=1, post_ms=1):
        gr.sync_block.__init__(self,
            name="ring_buffer",
            in_sig=[np.complex64],
            out_sig=[])

        self.samp_rate = samp_rate
        self.buffer_len = int(samp_rate * buffer_ms / 1000)
        self.post_len = int(samp_rate * post_ms / 1000)
        self.buf = deque(maxlen=self.buffer_len)
        self.trigger = False
        self.writer = None
        self.center_freq = None
        self.carrier_hz = None

    def set_trigger(self, center_freq=None, carrier_hz=None):
        self.trigger = True
        # Always update the current center and carrier frequencies
        if center_freq is not None:
            self.center_freq = center_freq
        if carrier_hz is not None:
            self.carrier_hz = carrier_hz
        # Add timestamp in format YYYYMMDDTHHMMSS
        timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        # Use the most up-to-date values for filename
        if self.center_freq is not None and self.carrier_hz is not None:
            self._current_filename = f"Center_{self.center_freq/1e6:.3f}MHz_Carrier_{self.carrier_hz/1e6:.3f}MHz_{timestamp}.iq"
        else:
            self._current_filename = f"capture_{timestamp}.iq"
        self.writer = open(self._current_filename, "wb")

    def work(self, input_items, output_items):
        data = input_items[0]

        # luôn ghi vào ring-buffer
        self.buf.extend(data)

        if self.trigger:
            # ghi pre-trigger (chỉ 1 lần)
            try:
                # if len(self.buf) >= self.buffer_len:
                if len(self.buf) > 0:
                    arr = np.array(self.buf, dtype=np.complex64)
                    self.writer.write(arr.tobytes())
                    self.buf.clear()
            except Exception:
                pass

            # ghi post-trigger
            try:
                self.writer.write(data.tobytes())
                self.post_len -= len(data)
            except Exception:
                pass

            if self.post_len <= 0:
                self.writer.close()
                # Check if file is empty and delete if so
                self._delete_if_empty(self._current_filename)
                self.trigger = False

        return len(data)