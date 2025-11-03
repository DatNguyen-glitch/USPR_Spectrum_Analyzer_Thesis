from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import blocks
import numpy
from gnuradio import digital
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import filter
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
import signal_detection_swep_cent_freq as swep_cent_freq  # embedded python module
import sip
import csv, threading
import numpy as np



class SignalDetector(gr.sync_block):
    """
    SignalDetector block (input: vector float dB)
    Logs detections to CSV and prints summary.
    """
    def __init__(self,
                 vec_len=2048,
                 samp_rate=2.4e6,
                 center_freq=4.5e6,
                 margin_db=6.0,
                 min_bw_hz=1e3,
                 ignore_center_bins=3,
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
                clusters.append((i, j))
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
                print(f"[SignalDetector] DETECT @ {carrier_hz/1e6:.6f} MHz | BW={bw_hz:.1f} Hz | peak={peak_db:.2f} dB | noise={noise_floor_db:.2f} dB | SNR={snr_db:.2f} dB")

                with open(self.csvfile, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([tnow, carrier_hz, bw_hz, peak_db, noise_floor_db, snr_db])

                self._consec_count = 0

        return len(input_items[0])

class signal_detection(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "signal_detection")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.overlap = overlap = 1e-1
        self.chunk_bw = chunk_bw = 1e7
        self.step = step = chunk_bw*(1-overlap)
        self.vec_len = vec_len = 2048
        self.total_chunk = total_chunk = round( ((1e9-chunk_bw)/step) + 1 )
        self.samp_rate = samp_rate = 2.6e6
        self.qpsk = qpsk = digital.constellation_rect([-1-1j, -1+1j, 1+1j, 1-1j], [0, 1, 3, 2],
        4, 2, 2, 1, 1).base()
        self.noise = noise = 0
        self.gain_tx = gain_tx = 22
        self.gain_rx = gain_rx = 22
        self.fft_len = fft_len = 2048
        # self.cent_freq_source = cent_freq_source = swep_cent_freq.sweeper.next(step)
        self.cent_freq_source = cent_freq_source = 9e8
        self.cent_freq_sink = cent_freq_sink = 4.5e6
        # dwell time in milliseconds for each center frequency step
        self.sweep_dwell_ms = 1000
        # sweep enabled flag (toggle with checkbox)
        self.sweep_enabled = True

        ##################################################
        # Blocks
        ##################################################

        self._noise_range = qtgui.Range(0, 1, 1e-3, 0, 200)
        self._noise_win = qtgui.RangeWidget(self._noise_range, self.set_noise, "'noise'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._noise_win)
        self._gain_tx_range = qtgui.Range(0, 50, 5e-1, self.gain_tx, 200)
        self._gain_tx_win = qtgui.RangeWidget(self._gain_tx_range, self.set_gain_tx, "'gain_tx'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._gain_tx_win)
        self._gain_rx_range = qtgui.Range(0, 50, 5e-1, self.gain_rx, 200)
        self._gain_rx_win = qtgui.RangeWidget(self._gain_rx_range, self.set_gain_rx, "'gain_rx'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._gain_rx_win)
        self._cent_freq_sink_range = qtgui.Range(4.5e6, 1e9, step, 9e8, 200)
        self._cent_freq_sink_win = qtgui.RangeWidget(self._cent_freq_sink_range, self.set_cent_freq_sink, "'cent_freq_sink'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._cent_freq_sink_win)
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_source_0.set_center_freq(cent_freq_source, 0)
        self.uhd_usrp_source_0.set_antenna("RX2", 0)
        self.uhd_usrp_source_0.set_gain(gain_rx, 0)
        self.uhd_usrp_sink_0_0 = uhd.usrp_sink(
            ",".join(("", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            "",
        )
        self.uhd_usrp_sink_0_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_sink_0_0.set_center_freq(cent_freq_sink, 0)
        self.uhd_usrp_sink_0_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0_0.set_gain(gain_tx, 0)
        self.qtgui_vector_sink_f_0 = qtgui.vector_sink_f(
            vec_len,
            0,
            1.0,
            "x-Axis",
            "y-Axis",
            "",
            1, # Number of inputs
            None # parent
        )
        self.qtgui_vector_sink_f_0.set_update_time(0.10)
        self.qtgui_vector_sink_f_0.set_y_axis((-140), 10)
        self.qtgui_vector_sink_f_0.enable_autoscale(True)
        self.qtgui_vector_sink_f_0.enable_grid(False)
        self.qtgui_vector_sink_f_0.set_x_axis_units("")
        self.qtgui_vector_sink_f_0.set_y_axis_units("")
        self.qtgui_vector_sink_f_0.set_ref_level(0)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_vector_sink_f_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_vector_sink_f_0.set_line_label(i, labels[i])
            self.qtgui_vector_sink_f_0.set_line_width(i, widths[i])
            self.qtgui_vector_sink_f_0.set_line_color(i, colors[i])
            self.qtgui_vector_sink_f_0.set_line_alpha(i, alphas[i])

        self._qtgui_vector_sink_f_0_win = sip.wrapinstance(self.qtgui_vector_sink_f_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_vector_sink_f_0_win)
        self.qtgui_freq_sink_x_0_0 = qtgui.freq_sink_c(
            2048, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            cent_freq_sink, #fc
            samp_rate, #bw
            "", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0_0.enable_grid(False)
        self.qtgui_freq_sink_x_0_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_0_win)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            2048, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            cent_freq_source, #fc
            samp_rate, #bw
            "", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        self.fft_vxx_0 = fft.fft_vcc(fft_len, True, window.blackmanharris(fft_len), True, 10)
        self.digital_constellation_modulator_0_0 = digital.generic_mod(
            constellation=qpsk,
            differential=True,
            samples_per_symbol=4,
            pre_diff_code=True,
            excess_bw=0.35,
            verbose=False,
            log=False,
            truncate=False)
        # self.epy_block_0 = epy_block_0.blk(vec_len=2048, threshold_dB=10.0)  # embedded python block
        # instantiate SignalDetector and replace embedded block
        self.signal_detector = SignalDetector(vec_len=self.vec_len,
                                              samp_rate=self.samp_rate,
                                              center_freq=self.cent_freq_source,
                                              margin_db=10.0,
                                              min_bw_hz=1e3,
                                              ignore_center_bins=3,
                                              persistence_k=2,
                                              out_csv="detected_signals.csv")
        self.blocks_correctiq_0 = blocks.correctiq()
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, vec_len)
        self.blocks_stream_to_vector_1 = blocks.stream_to_vector(gr.sizeof_float*1, vec_len)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(10, vec_len, (-100))
        self.blocks_keep_one_in_n_0 = blocks.keep_one_in_n(gr.sizeof_gr_complex*vec_len, (round(samp_rate/fft_len/1000)))
        self.blocks_integrate_xx_0 = blocks.integrate_ff(100, vec_len)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(vec_len)
        self.blocks_add_xx_0_0 = blocks.add_vcc(1)
        self.blocks_multiply_xx_0 = blocks.multiply_vff(vec_len)
        self.analog_random_source_x_0_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 256, 10000))), True)
        self.analog_noise_source_x_0_0 = analog.noise_source_c(analog.GR_GAUSSIAN, noise, 36)
        self.analog_const_source_x_0 = analog.sig_source_f(0, analog.GR_CONST_WAVE, 0, 0, (1/100))


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_const_source_x_0, 0), (self.blocks_stream_to_vector_1, 0))
        self.connect((self.blocks_stream_to_vector_1, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_integrate_xx_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_nlog10_ff_0, 0))

        self.connect((self.analog_noise_source_x_0_0, 0), (self.blocks_add_xx_0_0, 1))
        self.connect((self.analog_random_source_x_0_0, 0), (self.digital_constellation_modulator_0_0, 0))
        self.connect((self.blocks_add_xx_0_0, 0), (self.qtgui_freq_sink_x_0_0, 0))
        self.connect((self.blocks_add_xx_0_0, 0), (self.uhd_usrp_sink_0_0, 0))
        self.connect((self.blocks_keep_one_in_n_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_integrate_xx_0, 0))
        ### Connect log10 block directly after log10
        # self.connect((self.blocks_integrate_xx_0, 0), (self.blocks_nlog10_ff_0, 0))
        # feed PSD (dB) to both detector and vector sink
        self.connect((self.blocks_nlog10_ff_0, 0), (self.signal_detector, 0))
        self.connect((self.blocks_nlog10_ff_0, 0), (self.qtgui_vector_sink_f_0, 0))
        # self.connect((self.epy_block_0, 0), (self.qtgui_vector_sink_f_0, 0))
        ### end median block connection
        self.connect((self.blocks_stream_to_vector_0, 0), (self.blocks_keep_one_in_n_0, 0))
        self.connect((self.digital_constellation_modulator_0_0, 0), (self.blocks_add_xx_0_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        # self.connect((self.uhd_usrp_source_0, 0), (self.dc_blocker_xx_0, 0))
        # self.connect((self.uhd_usrp_source_0, 0), (self.blocks_correctiq_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.qtgui_freq_sink_x_0, 0))

        # start a timer to step the center frequency (sweep)
        # self._sweep_timer = Qt.QTimer(self)
        # self._sweep_timer.timeout.connect(self._update_sweep_center_freq)
        # self._sweep_timer.start(self.sweep_dwell_ms)  # interval in ms; adjust as needed
        # Add a simple checkbox to enable/disable sweeping
        self._sweep_checkbox = Qt.QCheckBox("Enable sweep")
        self._sweep_checkbox.setChecked(self.sweep_enabled)
        self._sweep_checkbox.stateChanged.connect(lambda s: self.set_sweep_enabled(s == QtCore.Qt.Checked))
        self.top_layout.addWidget(self._sweep_checkbox)

        self._sweep_timer = Qt.QTimer(self)
        self._sweep_timer.timeout.connect(self._update_sweep_center_freq)
        if self.sweep_enabled:
            self._sweep_timer.start(self.sweep_dwell_ms)  # interval in ms; adjust as needed

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "signal_detection")
        self.settings.setValue("geometry", self.saveGeometry())
        try:
            if hasattr(self, "_sweep_timer") and self._sweep_timer.isActive():
                self._sweep_timer.stop()
        except Exception:
            pass
        self.stop()
        self.wait()

        event.accept()

    def get_overlap(self):
        return self.overlap

    def set_overlap(self, overlap):
        self.overlap = overlap
        self.set_step(self.chunk_bw*(1-self.overlap))

    def get_chunk_bw(self):
        return self.chunk_bw

    def set_chunk_bw(self, chunk_bw):
        self.chunk_bw = chunk_bw
        self.set_step(self.chunk_bw*(1-self.overlap))
        self.set_total_chunk(round( ((1e9-self.chunk_bw)/self.step) + 1 ))

    def get_step(self):
        return self.step

    def set_step(self, step):
        self.step = step
        self.set_cent_freq_source(swep_cent_freq.sweeper.next(self.step))
        self.set_total_chunk(round( ((1e9-self.chunk_bw)/self.step) + 1 ))

    def get_vec_len(self):
        return self.vec_len

    def set_vec_len(self, vec_len):
        self.vec_len = vec_len
        if hasattr(self, 'signal_detector'):
            try:
                self.signal_detector.set_vec_len(self.vec_len)
            except Exception:
                pass

    def get_total_chunk(self):
        return self.total_chunk

    def set_total_chunk(self, total_chunk):
        self.total_chunk = total_chunk

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_keep_one_in_n_0.set_n((round(self.samp_rate/self.fft_len/1000)))
        self.qtgui_freq_sink_x_0.set_frequency_range(self.cent_freq_source, self.samp_rate)
        self.qtgui_freq_sink_x_0_0.set_frequency_range(self.cent_freq_sink, self.samp_rate)
        self.uhd_usrp_sink_0_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        if hasattr(self, 'signal_detector'):
            try:
                self.signal_detector.set_samp_rate(self.samp_rate)
            except Exception:
                pass

    def get_qpsk(self):
        return self.qpsk

    def set_qpsk(self, qpsk):
        self.qpsk = qpsk

    def get_noise(self):
        return self.noise

    def set_noise(self, noise):
        self.noise = noise
        self.analog_noise_source_x_0_0.set_amplitude(self.noise)

    def get_gain_tx(self):
        return self.gain_tx

    def set_gain_tx(self, gain_tx):
        self.gain_tx = gain_tx
        self.uhd_usrp_sink_0_0.set_gain(self.gain_tx, 0)

    def get_gain_rx(self):
        return self.gain_rx

    def set_gain_rx(self, gain_rx):
        self.gain_rx = gain_rx
        self.uhd_usrp_source_0.set_gain(self.gain_rx, 0)

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.blocks_keep_one_in_n_0.set_n((round(self.samp_rate/self.fft_len/1000)))

    def get_cent_freq_source(self):
        return self.cent_freq_source

    def set_cent_freq_source(self, cent_freq_source):
        self.cent_freq_source = cent_freq_source
        self.qtgui_freq_sink_x_0.set_frequency_range(self.cent_freq_source, self.samp_rate)
        self.uhd_usrp_source_0.set_center_freq(self.cent_freq_source, 0)
        if hasattr(self, 'signal_detector'):
            try:
                self.signal_detector.set_center_freq(self.cent_freq_source)
            except Exception:
                pass

    def get_cent_freq_sink(self):
        return self.cent_freq_sink

    def set_cent_freq_sink(self, cent_freq_sink):
        self.cent_freq_sink = cent_freq_sink
        self.qtgui_freq_sink_x_0_0.set_frequency_range(self.cent_freq_sink, self.samp_rate)
        self.uhd_usrp_sink_0_0.set_center_freq(self.cent_freq_sink, 0)

    def _update_sweep_center_freq(self):
        """Advance sweeper by one step and apply to source/sink. Restarts at end."""
        try:
            next_freq = swep_cent_freq.sweeper.next(self.step)
            if next_freq is None:
                # restart sweep
                swep_cent_freq.sweeper.chunk_index = 0
                next_freq = swep_cent_freq.sweeper.next(self.step)
            if next_freq is not None:
                self.set_cent_freq_source(next_freq)
                print(f"Sweeper: Setting center frequency to {next_freq/1e6} MHz", file=sys.stderr)
        except Exception as e:
            # swallow exceptions from sweeper to avoid timer crash
            print(f"Sweep update error: {e}", file=sys.stderr)

    def get_sweep_enabled(self):
        return self.sweep_enabled

    def set_sweep_enabled(self, enabled):
        """Enable or disable the sweep. Starts/stops the QTimer accordingly."""
        self.sweep_enabled = bool(enabled)
        try:
            if self.sweep_enabled:
                if not self._sweep_timer.isActive():
                    self._sweep_timer.start(self.sweep_dwell_ms)
            else:
                if self._sweep_timer.isActive():
                    self._sweep_timer.stop()
        except Exception:
            pass


def main(top_block_cls=signal_detection, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
