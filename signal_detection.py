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
from signal_detector import SignalDetector # embedded python module
from ring_buffer import ring_buffer
import sip
import csv, threading
import numpy as np


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
        self.chunk_bw = chunk_bw = 5e7
        self.step = step = chunk_bw*(1-overlap)
        self.vec_len = vec_len = 2048
        self.total_chunk = total_chunk = round( ((1e9-chunk_bw)/step) + 1 )
        self.samp_rate = samp_rate = 5e7
        self.variable_low_pass_filter_taps_0 = variable_low_pass_filter_taps_0 = firdes.low_pass(1.0, samp_rate, 2.25e7,1e6, window.WIN_HAMMING, 6.76)
        self.qpsk = qpsk = digital.constellation_rect([-1-1j, -1+1j, 1+1j, 1-1j], [0, 1, 3, 2],
        4, 2, 2, 1, 1).base()
        self.noise = noise = 0
        self.gain_tx = gain_tx = 22
        self.gain_rx = gain_rx = 22
        self.fft_len = fft_len = 2048
        # self.cent_freq_source = cent_freq_source = swep_cent_freq.sweeper.next(step)
        self.cent_freq_source = cent_freq_source = 1e8
        ############## IMPORTANT: antenna must support this frequency! ##############
        self.cent_freq_sink = cent_freq_sink = 5.1e8   # IMPORTANT: antenna must support this frequency!
        #############################################################################

        # dwell time in milliseconds for each center frequency step
        self.sweep_dwell_ms = 30
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
        self._cent_freq_source_range = qtgui.Range(4.5e6, 1e9, step, 9e8, 200)
        self._cent_freq_source_win = qtgui.RangeWidget(self._cent_freq_source_range, self.set_cent_freq_source, "'cent_freq_source'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._cent_freq_source_win)
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(('serial=34D628E', 'lo_offset=6e6','num_recv_frames=32','recv_frame_size=8200')),
            uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc16",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_source_0.set_center_freq(cent_freq_source, 0)
        self.uhd_usrp_source_0.set_antenna("RX2", 0)
        self.uhd_usrp_source_0.set_gain(gain_rx, 0)
        self.uhd_usrp_source_0.set_auto_dc_offset(False, 0)
        self.uhd_usrp_source_0.set_auto_iq_balance(False, 0)
        self.uhd_usrp_sink_0_0 = uhd.usrp_sink(
            ",".join(('serial=34D62A7', 'serial=34D62A7,otw_format=sc16')),
            uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc16",
                args='',
                channels=list(range(0,1)),
            ),
            "",
        )
        self.uhd_usrp_sink_0_0.set_samp_rate(2e6)
        # No synchronization enforced.

        self.uhd_usrp_sink_0_0.set_center_freq(cent_freq_sink, 0)
        self.uhd_usrp_sink_0_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0_0.set_bandwidth(4e6, 0)
        self.uhd_usrp_source_0.set_rx_agc(False, 0)
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
        self.freq_xlating_fir_filter_xxx_0 = filter.freq_xlating_fir_filter_ccc(10, variable_low_pass_filter_taps_0, (-1e6), samp_rate)
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
        # instantiate ring_buffer and SignalDetector, and connect them
        self.ring_buffer = ring_buffer(samp_rate=self.samp_rate, buffer_ms=1, post_ms=1)
        self.signal_detector = SignalDetector(vec_len=self.vec_len,
                              samp_rate=self.samp_rate,
                              center_freq=self.cent_freq_source,
                              margin_db=30.0,
                              min_bw_hz=1e5,
                              ignore_center_bins=4,
                              persistence_k=4,
                              out_csv="detected_signals.csv",
                              ring_buffer=self.ring_buffer)
        self.dc_blocker_xx_0 = filter.dc_blocker_cc(10, False)
        self.blocks_correctiq_0 = blocks.correctiq()
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, vec_len)
        self.blocks_stream_to_vector_1 = blocks.stream_to_vector(gr.sizeof_float*1, vec_len)
        self.blocks_nlog10_ff_0 = blocks.nlog10_ff(10, vec_len, (-100))
        self.blocks_keep_one_in_n_0 = blocks.keep_one_in_n(gr.sizeof_gr_complex*vec_len, (round(samp_rate/fft_len/1000)))
        self.blocks_keep_one_in_n_1 = blocks.keep_one_in_n(gr.sizeof_float*vec_len, 5)
        self.blocks_integrate_xx_0 = blocks.integrate_ff(10, vec_len)
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(vec_len)
        self.blocks_add_xx_0_0 = blocks.add_vcc(1)
        self.blocks_multiply_xx_0 = blocks.multiply_vff(vec_len)
        self.analog_random_source_x_0_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 256, 10000))), True)
        self.analog_noise_source_x_0_0 = analog.noise_source_c(analog.GR_GAUSSIAN, noise, 36)
        self.analog_const_source_x_0 = analog.sig_source_f(0, analog.GR_CONST_WAVE, 0, 0, (1/10))
        self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=2,
                taps=[],
                fractional_bw=0)
        self.low_pass_filter_0 = filter.fir_filter_ccf(
            1,                              # decimation
            firdes.low_pass(
                1,                          # gain
                3e6,                        # sampling rate
                1.3e6,                      # cutoff frequency
                1e5,                        # transition width
                window.WIN_HAMMING,         # window type
                6.76))                      # beta
        self.low_pass_filter_1 = filter.fir_filter_ccf(
            1,                              # decimation
            firdes.low_pass(
                1,                          # gain
                5e7,                        # sampling rate
                22.5e6,                      # cutoff frequency
                1.5e6,                        # transition width
                window.WIN_HAMMING,         # window type
                6.76))                      # beta
        ##################################################
        # Connections
        ##################################################

        self.connect((self.analog_noise_source_x_0_0, 0), (self.blocks_add_xx_0_0, 1))
        self.connect((self.analog_random_source_x_0_0, 0), (self.digital_constellation_modulator_0_0, 0))
        self.connect((self.digital_constellation_modulator_0_0, 0), (self.blocks_add_xx_0_0, 0))
        self.connect((self.blocks_add_xx_0_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.uhd_usrp_sink_0_0, 0))
        # self.connect((self.low_pass_filter_0, 0), (self.qtgui_freq_sink_x_0_0, 0))        # visualize TX signal
        
        
        
        ### Connect log10 block directly after log10
        # self.connect((self.epy_block_0, 0), (self.qtgui_vector_sink_f_0, 0))
        ### end median block connection

        ####### ----------------- Signal Detection Chain ----------------- #######
        # self.connect((self.uhd_usrp_source_0, 0), (self.freq_xlating_fir_filter_xxx_0, 0))
        # self.connect((self.freq_xlating_fir_filter_xxx_0, 0), (self.dc_blocker_xx_0, 0))
        # self.connect((self.dc_blocker_xx_0, 0), (self.qtgui_freq_sink_x_0, 0))            # visualize RX signal
        # self.connect((self.dc_blocker_xx_0, 0), (self.blocks_stream_to_vector_0, 0))
        # # self.connect((self.blocks_stream_to_vector_0, 0), (self.blocks_keep_one_in_n_0, 0))
        # self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        # # self.connect((self.blocks_keep_one_in_n_0, 0), (self.fft_vxx_0, 0))
        # self.connect((self.fft_vxx_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        # self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_integrate_xx_0, 0))
        # self.connect((self.blocks_integrate_xx_0, 0), (self.blocks_multiply_xx_0, 0))
        # # Connect block to device output of integrator by 100
        # self.connect((self.analog_const_source_x_0, 0), (self.blocks_stream_to_vector_1, 0))
        # self.connect((self.blocks_stream_to_vector_1, 0), (self.blocks_multiply_xx_0, 1))
        # # After multiplication, connect to log10
        # self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_nlog10_ff_0, 0))
        # # feed PSD (dB) to both detector and vector sink
        # self.connect((self.blocks_nlog10_ff_0, 0), (self.signal_detector, 0))
        # self.connect((self.blocks_nlog10_ff_0, 0), (self.qtgui_vector_sink_f_0, 0))
        ####### ----------------- RAW Signal Detection Chain ----------------- #######
        self.connect((self.uhd_usrp_source_0, 0), (self.low_pass_filter_1, 0))
        self.connect((self.low_pass_filter_1, 0), (self.dc_blocker_xx_0, 0))
        # self.connect((self.dc_blocker_xx_0, 0), (self.qtgui_freq_sink_x_0, 0))            # visualize RX signal
        # Connect RX IQ stream to ring_buffer for event capture
        self.connect((self.dc_blocker_xx_0, 0), (self.ring_buffer, 0))
        self.connect((self.dc_blocker_xx_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_integrate_xx_0, 0))
        self.connect((self.blocks_integrate_xx_0, 0), (self.blocks_multiply_xx_0, 0))
        # Connect block to device output of integrator by 50
        self.connect((self.analog_const_source_x_0, 0), (self.blocks_stream_to_vector_1, 0))
        self.connect((self.blocks_stream_to_vector_1, 0), (self.blocks_multiply_xx_0, 1))
        # After multiplication, connect to log10
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_nlog10_ff_0, 0))
        # feed PSD (dB) to both detector and vector sink
        self.connect((self.blocks_nlog10_ff_0, 0), (self.signal_detector, 0))
        # self.connect((self.blocks_nlog10_ff_0, 0), (self.qtgui_vector_sink_f_0, 0))

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

    def get_variable_low_pass_filter_taps_0(self):
        return self.variable_low_pass_filter_taps_0

    def set_variable_low_pass_filter_taps_0(self, variable_low_pass_filter_taps_0):
        self.variable_low_pass_filter_taps_0 = variable_low_pass_filter_taps_0
        self.freq_xlating_fir_filter_xxx_0.set_taps(self.variable_low_pass_filter_taps_0)

    def get_total_chunk(self):
        return self.total_chunk

    def set_total_chunk(self, total_chunk):
        self.total_chunk = total_chunk

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_keep_one_in_n_0.set_n((round(self.samp_rate/self.fft_len/1000)))
        self.blocks_keep_one_in_n_1.set_n(5)
        self.qtgui_freq_sink_x_0.set_frequency_range(self.cent_freq_source, self.samp_rate)
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, 22.5e6, 1.5e6, window.WIN_HAMMING, 6.76))
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
                self.signal_detector.set_enabled(False)
                self.set_cent_freq_source(next_freq)
                print(f"Sweeper: Setting center frequency to {next_freq/1e6} MHz", file=sys.stderr)
                start_wait = time.time()
                timeout = 0.01 
                lo_locked = False
                while (time.time() - start_wait) < timeout:
                    lo_locked_sensor = self.uhd_usrp_source_0.get_sensor("lo_locked", 0)
                    if lo_locked_sensor.to_bool():
                        lo_locked = True
                        break
                    # time.sleep(0.001)
                after_wait = time.time()
                if lo_locked:
                    self.signal_detector.set_enabled(True)
                else:
                    print(f"WARNING: PLL failed to lock at {next_freq/1e6} MHz!", file=sys.stderr)
                    self.signal_detector.set_enabled(True)
        except Exception as e:
            # swallow exceptions from sweeper to avoid timer crash
            print(f"Sweep update error: {e}", file=sys.stderr)
            self.signal_detector.set_enabled(True)

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
