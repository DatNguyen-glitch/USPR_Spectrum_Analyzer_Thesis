import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIGURATION ---
WATCH_DIR = "./captures/2025-11-27"        # Directory to monitor
RESULT_DIR = "./results"        # Directory to save result images
SAMPLE_RATE = 50e6              # 50 MSps
FFT_SIZE = 65536                # High resolution (64k points)
SCALING_FACTOR = 131072.0       # Scaling factor (must match the multiplier in ring_buffer)

# Backend 'Agg' to run in background, no popup window
plt.switch_backend('Agg')

# Ensure directories exist
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(WATCH_DIR):
    os.makedirs(WATCH_DIR)

class NewIQFileHandler(FileSystemEventHandler):
    """
    Event handler class for the OS file system events.
    Only reacts when a file has been written and closed (on_closed).
    """
    def on_closed(self, event):
        # Only process files, not directories
        if event.is_directory:
            return

        # Only process .iq files
        filename = os.path.basename(event.src_path)
        if not filename.endswith(".iq"):
            return

        print(f"[Event] New file completed: {filename}")
        
        # Gọi hàm xử lý
        process_and_cleanup(event.src_path, filename)

def process_and_cleanup(filepath, filename):
    try:
        # 1. Extract center frequency from filename (Parse Filename)
        # Assumed format: ...Fc520000000...
        center_freq = 0.0
        try:
            # Find text starting with Fc followed by numbers
            import re
            match = re.search(r'Fc(\d+)', filename)
            if match:
                center_freq = float(match.group(1))
        except:
            print("    -> Warning: Could not read Fc from filename, defaulting to 0Hz")

        # 2. Read IQ data (Int16)
        # Use memmap for optimized read speed
        raw_data = np.fromfile(filepath, dtype=np.int16)
        
        if len(raw_data) < FFT_SIZE * 2:
            print("    -> File too short or corrupted. Deleting.")
            os.remove(filepath)
            return

        # 3. Convert to Complex Float
        # Separate I/Q
        i_data = raw_data[0::2].astype(np.float32)
        q_data = raw_data[1::2].astype(np.float32)
        complex_data = (i_data + 1j * q_data) / SCALING_FACTOR

        # 4. High-order FFT Analysis (Welch Method)
        print(f"    -> Computing {FFT_SIZE}-point FFT...")
        freqs, psd = signal.welch(
            complex_data,
            fs=SAMPLE_RATE,
            window='blackmanharris',
            nperseg=FFT_SIZE,
            return_onesided=False
        )

        # Shift and convert to dB
        freqs = np.fft.fftshift(freqs)
        psd_db = 10 * np.log10(np.fft.fftshift(psd) + 1e-15)
        
        # Actual frequencies
        real_freqs_mhz = (freqs + center_freq) / 1e6

        # Find peak
        peak_idx = np.argmax(psd_db)
        peak_freq = real_freqs_mhz[peak_idx]
        peak_pwr = psd_db[peak_idx]

        # 5. Plot and save image
        plt.figure(figsize=(10, 5))
        plt.plot(real_freqs_mhz, psd_db, color='#004488', linewidth=0.6)
        plt.title(f"Spectrum Analysis: {filename}\nPeak: {peak_freq:.4f} MHz ({peak_pwr:.2f} dB)")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude (dB)")
        plt.grid(True, alpha=0.5)
        
        # Mark the peak
        plt.plot(peak_freq, peak_pwr, 'rx')
        
        # Save image
        out_png = os.path.join(RESULT_DIR, filename.replace(".iq", ".png"))
        plt.savefig(out_png, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"    -> Image saved: {out_png}")

        # 6. DELETE ORIGINAL FILE (Cleanup)
        os.remove(filepath)
        print(f"    -> Original IQ file deleted.")

    except Exception as e:
        print(f"    -> ERROR: {e}")
        # Optionally delete corrupted file to avoid congestion
        # os.remove(filepath) 

if __name__ == "__main__":
    path = WATCH_DIR
    print(f"Monitor folder: {os.path.abspath(path)}")

    # ==========================================
    # Step 1: Scan and process any existing .iq files
    # ==========================================
    print(f"[*] Scanning for backlog files...")
    existing_files = [f for f in os.listdir(path) if f.endswith(".iq")]
    
    if existing_files:
        print(f" -> Found {len(existing_files)} old files. Processing...")
        for fname in existing_files:
            fpath = os.path.join(path, fname)
            # Reuse existing processing function
            process_and_cleanup(fpath, fname)
        print(f"[*] Backlog files cleaned up.")
    else:
        print(f" -> No backlog files found.")

    # ==========================================
    # Step 2: Start monitoring for new .iq files
    # ==========================================
    event_handler = NewIQFileHandler()
    observer = Observer()
    
    # Register recursive directory monitoring (recursive=True to also watch date subdirectories)
    observer.schedule(event_handler, path, recursive=True)
    
    print(f"--- WATCHDOG SERVICE STARTED ---")
    print(f"Monitoring directory: {os.path.abspath(path)}")
    print(f"Press Ctrl+C to stop.")
    
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()