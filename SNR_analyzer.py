import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt

GAIN_TX = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]   # các điểm đo sạch sau pre-scan
# GAIN_TX = [383, 433, 483, 508] 
# GAIN_TX = [793, 818, 868, 918, 943]

TARGET_FREQ_MHZ = 433.5    # tần số tín hiệu test cố định (MHz)

power_at_gain = []
noise_floor_at_gain = []

for gain_tx in GAIN_TX:
    path_pattern = f'/home/datnguyen/Desktop/USPR_Spectrum_Analyzer_Thesis/spectrum_data/Jun14/{TARGET_FREQ_MHZ}Mhz_RX45TX{gain_tx}_open_SNR/fft_*'
    file_list = glob.glob(path_pattern)    
    with h5py.File(file_list[0], 'r') as hf:
        db_min = hf.attrs.get("db_min", -120.0)  # -120.0
        db_max = hf.attrs.get("db_max", 0.0)   #    0.0

        fft_uint8 = hf['fft_data'][:]          # shape (N_frames, fft_len)
        center_hz = float(hf['center_hz'][0])
        sample_rate = float(hf.attrs.get("sample_rate", 50e6))

        # Dequantize (result: dB)
        fft_db = fft_uint8 / 255.0 * (db_max - db_min) + db_min

        # Determine fft length and frequency axis
        fft_len = fft_db.shape[1]
        freq_axis = center_hz + np.fft.fftshift(np.fft.fftfreq(fft_len, 1.0 / sample_rate))

        # Suy ra target_bin và window từ TARGET_FREQ_MHZ
        # Fs=50MHz, N=2048 -> bin_resolution = 24.4 kHz/bin
        # Chon +-5 bins (~+-122 kHz): du rong de bat CFO nho, du hep de tranh interference
        bin_resolution_hz = sample_rate / fft_len
        target_bin        = int(np.argmin(np.abs(freq_axis - TARGET_FREQ_MHZ * 1e6)))
        half_window       = 5

        # Voi moi frame: tim bin dinh trong window -> doc power tai dung bin do
        peaks = []
        peak_freqs_hz = []
        for frame_db in fft_db:
            lo = max(1,           target_bin - half_window)
            hi = min(fft_len - 1, target_bin + half_window + 1)
            if lo >= hi:
                continue
            peak_bin_local = int(np.argmax(frame_db[lo:hi]))
            peak_bin       = lo + peak_bin_local
            peaks.append(float(frame_db[peak_bin]))
            peak_freqs_hz.append(freq_axis[peak_bin])

        if len(peaks) == 0:
            peak_power = float('nan')
        else:
            peak_power   = float(np.mean(peaks))
            mean_peak_hz = float(np.mean(peak_freqs_hz))
            print(f"[TX gain={gain_tx:>2} dB]  target={TARGET_FREQ_MHZ:.3f} MHz  |  "
                  f"peak found={mean_peak_hz/1e6:.4f} MHz  |  "
                  f"offset={( mean_peak_hz - TARGET_FREQ_MHZ*1e6)/1e3:+.2f} kHz  |  "
                  f"power={peak_power:.2f} dBFS")

        power_at_gain.append(peak_power)
        noise_floor_at_gain.append(float(np.median(fft_db)))

snr = np.array(power_at_gain) - np.array(noise_floor_at_gain)
# Vẽ
plt.figure(figsize=(10, 5))
# plt.ylim([-65, -45])
plt.plot(GAIN_TX, power_at_gain, 'o-', linewidth=1.5, markersize=5, label="peak signal")
plt.plot(GAIN_TX, snr, 's-', linewidth=1.5, markersize=5, label="SNR (dB)")
plt.plot(GAIN_TX, noise_floor_at_gain, 's-', linewidth=1.5, markersize=5, label="Noise Floor (dB)")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Transmit Gain (dB)')
plt.ylabel('SNR (dB)')
plt.title(f'SNR — {TARGET_FREQ_MHZ} MHz Band')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig(f'snr_{TARGET_FREQ_MHZ}MHz.png', dpi=300)
plt.show()