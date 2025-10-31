"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import threading
import numpy as np
from gnuradio import gr

class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const"""

    def __init__(self, vec_len=2048):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(self,
                               name='MedianProbe',
                               in_sig=[(np.float32, vec_len)],
                               out_sig=[(np.float32, vec_len)])
        self.vec_len = int(vec_len)
        self._lock = threading.Lock()
        self._latest = None
        
    def work(self, input_items, output_items):
        inp = input_items[0]
        out = output_items[0]
        n_items = len(inp)
        for i in range(n_items):
            v = np.asarray(inp[i], dtype=np.float32)
            m = float(np.median(v))
            with self._lock:
                self._latest = m
            print(f"Median: {m} dB", flush=True)
            out[i][:] = v
        return n_items

    def get_latest(self):
        """Return the most recent median (float) or None if not available."""
        with self._lock:
            return self._latest
