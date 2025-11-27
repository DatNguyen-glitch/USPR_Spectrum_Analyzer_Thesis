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
        self.current_filename = ""
        self.post_counter = 0

    def set_trigger(self, center_freq=None, carrier_hz=None):
        if self.trigger:
            return  # Already triggered, ignore new trigger requests
        
        self.trigger = True
        self.post_counter = self.post_len

        now = time.localtime()
        date_str = time.strftime("%Y-%m-%d", now)
        time_str = time.strftime("%H-%M-%S", now)
        base_dir = "./captures"
        target_dir = os.path.join(base_dir, date_str)
        if not os.path.exists(target_dir):
            try:
                os.makedirs(target_dir)
            except OSError:
                pass
        filename = f"CAP_{time_str}_Fc{center_freq/1e6:.3f}_Fs{carrier_hz/1e6:.3f}.iq"
        self.current_filename = os.path.join(target_dir, filename)
        try:
            self.writer = open(self.current_filename, "wb")
            # Write pre-trigger buffer
            if len(self.buf) > 0:
                arr = np.array(self.buf, dtype=np.complex64)
                arr_int16 = (arr.view(np.float32) * 32767*4).clip(-32768*4, 32767*4).astype(np.int16)
                self.writer.write(arr_int16.tobytes())

        # # Always update the current center and carrier frequencies
        # if center_freq is not None:
        #     self.center_freq = center_freq
        # if carrier_hz is not None:
        #     self.carrier_hz = carrier_hz
        # # Add timestamp in format YYYYMMDDTHHMMSS
        # timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        # # Use the most up-to-date values for filename
        # if self.center_freq is not None and self.carrier_hz is not None:
        #     self.current_filename = f"Center_{self.center_freq/1e6:.3f}MHz_Carrier_{self.carrier_hz/1e6:.3f}MHz_{timestamp}.iq"
        # else:
        #     self.current_filename = f"capture_{timestamp}.iq"

        # try:
        #     self.writer = open(self.current_filename, "wb")
        #     if len(self.buf) > 0:
        #         arr = np.array(self.buf, dtype=np.complex64)
        #         self.writer.write(arr.tobytes())
        #     return self.current_filename
        except Exception as e:
            print(f"RingBuffer Write Error: {e}")
            self.trigger = False
            self.current_filename = ""

    def work(self, input_items, output_items):
        data = input_items[0]
        # Always update ring-buffer
        self.buf.extend(data)

        if self.trigger and self.writer:
            try:
                # if len(self.buf) >= self.buffer_len:
                n_write = min(len(data), self.post_counter)
                
                if n_write > 0:
                    chunk = data[:n_write]
                    chunk_int16 = (chunk.view(np.float32) * 32767*4).clip(-32768*4, 32767*4).astype(np.int16)
                    self.writer.write(chunk_int16.tobytes())
                    self.post_counter -= n_write

                if self.post_counter <= 0:
                    self.writer.close()
                    self.writer = None
                    self.trigger = False
                    if os.path.getsize(self.current_filename) == 0:
                        os.remove(self.current_filename)
            except Exception:
                print(f"RingBuffer Work Error: {e}")
                self.trigger = False
                if self.writer: self.writer.close()
        return len(data)