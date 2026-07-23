import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt

FREQS_MHZ = [358.5, 366, 373.5, 381, 388.5, 396, 403.5, 411, 
             418.5, 426, 433.5, 441, 448.5, 456, 463.5, 471, 478.5, 486, 493.5, 501]   # các điểm đo sạch sau pre-scan
# FREQS_MHZ = [793, 800.5, 808, 815.5, 823, 830.5, 838, 845.5, 
#              853, 860.5, 868, 875.5, 883, 890.5, 898, 905.5, 913, 920.5, 928, 935.5]
power_at_freq = []
noise_floor_at_gain = []

for f_mhz in FREQS_MHZ:
    path_pattern = f'/home/datnguyen/Desktop/USPR_Spectrum_Analyzer_Thesis/spectrum_data/Jun14/{f_mhz}Mhz_RX45TX75_open_FRC/fft_*'
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

        # Tính bin resolution và suy ra window từ tần số mục tiêu
        # Fs=50MHz, N=2048 -> bin_resolution = 24.4 kHz/bin
        # Chon +-5 bins (~+-122 kHz): du rong de bat CFO nho, du hep de tranh interference
        bin_resolution_hz = sample_rate / fft_len
        target_bin        = int(np.argmin(np.abs(freq_axis - f_mhz * 1e6)))
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
            peak_power    = float(np.mean(peaks))
            mean_peak_hz  = float(np.mean(peak_freqs_hz))
            print(f"[{f_mhz:>6.1f} MHz]  target={f_mhz:.3f} MHz  |  "
                  f"peak found={mean_peak_hz/1e6:.4f} MHz  |  "
                  f"offset={( mean_peak_hz - f_mhz*1e6)/1e3:+.2f} kHz  |  "
                  f"power={peak_power:.2f} dBFS")

        power_at_freq.append(peak_power)
        noise_floor_at_gain.append(float(np.median(fft_db)))


# Normalize về 0 dB tại điểm mạnh nhất (hoặc tại 433.5 MHz)
ref_idx = FREQS_MHZ.index(433.5)
# power_normalized = np.array(power_at_freq) - power_at_freq[ref_idx]

# Vẽ
plt.figure(figsize=(10, 5))
# plt.ylim([-3, 3])
plt.plot(FREQS_MHZ, power_at_freq, 'o-', linewidth=1.5, markersize=5, label="Uncalibrated Rx Power")
plt.plot(FREQS_MHZ, noise_floor_at_gain, 's-', linewidth=1.5, markersize=5, label="Noise Floor")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Relative Power (dB)')
plt.title('System Frequency Response (Over-The-Air 50 meter) — 433 MHz Band')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig('freq_response_433MHz.png', dpi=300)
plt.show()