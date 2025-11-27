from gnuradio import gr
from metadata_logger import MetadataLogger
import numpy as np
import csv
import threading
import time
import os
import pmt

class SignalDetector(gr.sync_block):
    """
    SignalDetector block (input: vector float dB)
    Logs detections to CSV and prints summary.
    """
    def __init__(self,
                 vec_len=2048,
                 samp_rate=5e7,
                 center_freq=2.5e7,
                 margin_db=30.0,
                 min_bw_hz=1e5,
                 ignore_center_bins=4,
                 persistence_k=4,
                 out_csv="detected_signals.csv",
                 ring_buffer=None):
        gr.sync_block.__init__(self,
            name="SignalDetector",
            in_sig=[(np.float32, int(vec_len))],
            out_sig=[]
        )
        self.ring_buffer = ring_buffer
        self.N = int(vec_len)
        self.fs = float(samp_rate)
        self.center_freq = float(center_freq)       # auto update through Tag
        self.margin_db = float(margin_db)
        self.min_bw_hz = float(min_bw_hz)
        self.ignore_center = int(ignore_center_bins)
        self.persistence_k = int(persistence_k)

        self.df = self.fs / self.N
        self.min_bins = max(1, int(np.ceil(self.min_bw_hz / self.df)))
        self._lock = threading.Lock()
        self._consec_count = 0

        self.csvfile = out_csv
        self.metadata_db = MetadataLogger("signals_metadata.db")
        try:
            self._csv_file_handle = open(self.csvfile, 'a', newline='')
            self._csv_writer = csv.writer(self._csv_file_handle)
            if self._csv_file_handle.tell() == 0:
                 self._csv_writer.writerow(["time", "carrier_Hz", "bandwidth_Hz", "peak_dB", "noise_floor_dB", "snr_dB"])
                 self._csv_file_handle.flush()
        except Exception as e:
            print(f"[SignalDetector] CSV Error: {e}")
            self._csv_file_handle = None

        self._blanking_samples = 0
        self.BLANK_VEC_COUNT = 50
        self.enabled = True
        # Hanning Window 5 tap or Moving Average smoothing the noise
        self.smooth_window = np.ones(5) / 5.0

    def set_center_freq(self, f_hz):
        pass
        # with self._lock:
        #     self.center_freq = float(f_hz)
        #     self._consec_count = 0
        #     self._blanking_samples = 10         # blank for 10 samples after freq change

    def set_samp_rate(self, samp_rate):
        with self._lock:
            self.fs = float(samp_rate)
            self.df = self.fs / self.N
            self.min_bins = max(1, int(np.ceil(self.min_bw_hz / self.df)))

    def set_vec_len(self, vec_len):
        with self._lock:
            self.N = int(vec_len)
            self.df = self.fs / self.N
            self.min_bins = max(1, int(np.ceil(self.min_bw_hz / self.df)))

    def estimate_noise_median(self, psd_db):
        return float(np.median(psd_db))

    def find_clusters(self, mask):
        clusters = []
        N = len(mask)
        i = 0
        while i < N:
            if mask[i]:
                j = i
                while j+1 < N and mask[j+1]:
                    j += 1
                clusters.append((i, j)) # i = start of a cluster, j = end of a cluster
                i = j+1
            else:
                i += 1
        return clusters

    def compute_freq_for_bin(self, k):
        return self.center_freq - (self.fs/2.0) + k * self.df

    def set_enabled(self, state):
        self.enabled = state
        if not state:
            self._consec_count = 0

    def work(self, input_items, output_items):
        if not self.enabled:
            return len(input_items[0])
        invecs = input_items[0]
        n_items = len(invecs)
        
        tags = self.get_tags_in_window(0, 0, n_items)
        for tag in tags:
            key = pmt.to_python(tag.key)
            if key == 'rx_freq':
                new_freq = float(pmt.to_python(tag.value))
                if abs(new_freq - self.center_freq) > 1.0:
                    print(f"[Sync] Freq changed to {new_freq/1e6:.1f} MHz based on Tag")
                    self.center_freq = new_freq
                    self.center_freq = new_freq
                    self._consec_count = 0
                    self._blanking_samples = self.BLANK_VEC_COUNT

        for i in range(n_items):
            if self._blanking_samples > 0:
                self._blanking_samples -= 1     # blank for 10 samples after freq change
                continue

            raw_psd_db = invecs[i]
            psd_db = np.convolve(raw_psd_db, self.smooth_window, mode='same')
            psd_db = np.array(psd_db, dtype=np.float32)

            if self.ignore_center > 0:
                center = self.N // 2
                low = max(0, center - self.ignore_center)
                high = min(self.N - 1, center + self.ignore_center)
                psd_mask_for_noise = np.concatenate((psd_db[:low], psd_db[high+1:]))
            else:
                psd_mask_for_noise = psd_db

            noise_floor_db = self.estimate_noise_median(psd_mask_for_noise)
            threshold_db = noise_floor_db + self.margin_db
            mask = psd_db > threshold_db

            if self.ignore_center > 0:
                mask[low:high+1] = False

            clusters = self.find_clusters(mask)

            detected_any = False
            best_cluster = None
            best_peak_db = -9999.0
            for (s,e) in clusters:
                width_bins = e - s + 1
                if width_bins < self.min_bins:
                    continue
                detected_any = True
                local_peak_idx = s + int(np.argmax(psd_db[s:e+1]))
                peak_db = float(psd_db[local_peak_idx])
                if peak_db > best_peak_db:
                    best_peak_db = peak_db
                    best_cluster = (s,e,local_peak_idx,peak_db)

            if detected_any:
                self._consec_count += 1
            else:
                self._consec_count = 0

            if self._consec_count >= self.persistence_k and best_cluster is not None:
                s,e,peak_idx,peak_db = best_cluster
                bw_hz = (e - s + 1) * self.df
                carrier_hz = self.compute_freq_for_bin(peak_idx)
                snr_db = peak_db - noise_floor_db

                tnow = time.time()
                # print(f"[SignalDetector] DETECT @ {carrier_hz/1e6:.6f} MHz "
                #       f"| BW={bw_hz/1e6:.6f} MHz | peak={peak_db:.2f} dB | noise={noise_floor_db:.2f} dB "
                #       f"| SNR={snr_db:.2f} dB")
                # if self._csv_file_handle is not None:
                #     self._csv_writer.writerow([tnow, f"{carrier_hz/1e6:.6f}", f"{bw_hz/1e6:.6f}", f"{peak_db:.2f}", f"{noise_floor_db:.2f}", f"{snr_db:.2f}"])
                #     self._csv_file_handle.flush()
                detected_filename = ""
                if self.ring_buffer:
                    self.ring_buffer.set_trigger(center_freq=self.center_freq, carrier_hz=carrier_hz)
                    detected_filename = self.ring_buffer.current_filename
                print(f"[SignalDetector] DETECT @ {carrier_hz/1e6:.6f} MHz | File: {os.path.basename(detected_filename)}", flush=True)
                # --- METADATA CAPTURE ---
                if hasattr(self, 'metadata_db'):
                    self.metadata_db.log_capture(
                        filename=detected_filename,
                        timestamp=tnow,
                        freq=carrier_hz,
                        bw=bw_hz,
                        peak=peak_db,
                        snr=snr_db,
                        duration=0.0
                )
                # Trigger ring buffer if available
                # if self.ring_buffer is not None:
                #     try:
                #         self.ring_buffer.set_trigger(center_freq=self.center_freq,carrier_hz=carrier_hz)
                #     except Exception as e:
                #         print(f"[SignalDetector] Failed to trigger ring buffer: {e}")
                self._consec_count = 0

        return len(input_items[0])
