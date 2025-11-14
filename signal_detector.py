from gnuradio import gr
import numpy as np
import csv
import threading
import time


class SignalDetector(gr.sync_block):
    """
    SignalDetector block (input: vector float dB)
    Logs detections to CSV and prints summary.
    """
    def __init__(self,
                 vec_len=4096,
                 samp_rate=5e7,
                 center_freq=2.25e7,
                 margin_db=6.0,
                 min_bw_hz=1e5,
                 ignore_center_bins=0,
                 persistence_k=2,
                 out_csv="detected_signals.csv"):
        gr.sync_block.__init__(self,
            name="SignalDetector",
            in_sig=[(np.float32, int(vec_len))],
            out_sig=[]
        )
        self.N = int(vec_len)
        self.fs = float(samp_rate)
        self.center_freq = float(center_freq)
        self.margin_db = float(margin_db)
        self.min_bw_hz = float(min_bw_hz)
        self.ignore_center = int(ignore_center_bins)
        self.persistence_k = int(persistence_k)

        self.df = self.fs / self.N
        self.min_bins = max(1, int(np.ceil(self.min_bw_hz / self.df)))
        self._lock = threading.Lock()
        self._consec_count = 0

        self.csvfile = out_csv
        try:
            with open(self.csvfile, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["time", "carrier_Hz", "bandwidth_Hz", "peak_dB", "noise_floor_dB", "snr_dB"])
        except FileExistsError:
            print("Failed to create CSV file; it may already exist.", flush=True)
            pass

    def set_center_freq(self, f_hz):
        with self._lock:
            self.center_freq = float(f_hz)

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

    def work(self, input_items, output_items):
        invecs = input_items[0]
        for psd_db in invecs:
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
                print(f"[SignalDetector] DETECT @ {carrier_hz/1e6:.6f} MHz "
                      f"| BW={bw_hz:.1f} Hz | peak={peak_db:.2f} dB | noise={noise_floor_db:.2f} dB "
                      f"| SNR={snr_db:.2f} dB")

                with open(self.csvfile, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([tnow, f"{carrier_hz/1e6:.6f}", bw_hz, f"{peak_db:.2f}", f"{noise_floor_db:.2f}", f"{snr_db:.2f}"])

                self._consec_count = 0

        return len(input_items[0])
