import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIGURATION ---
WATCH_DIR = "./captures"        # Base captures directory (contains date subfolders)
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
        
        # Invoke processing function
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

        # --- Enhanced Time-Domain Plot ---
        # 1. Compute magnitude to locate the signal
        magnitude = np.abs(complex_data)
        
        # 2. Find the index of the highest peak (likely pulse location)
        peak_time_idx = np.argmax(magnitude)
        
        # 3. Select display window: 100 samples before and 400 samples after the peak
        start_plot = max(0, peak_time_idx - 100)
        end_plot = min(len(complex_data), peak_time_idx + 400)
        
        # Only plot if a clear peak is found (avoid plotting pure noise)
        if end_plot > start_plot:
            plt.figure(figsize=(10, 4))
            
            # Plot Real (I) and Imag (Q) components
            plt.plot(range(start_plot, end_plot), np.real(complex_data[start_plot:end_plot]), label='Real (I)', linewidth=1)
            plt.plot(range(start_plot, end_plot), np.imag(complex_data[start_plot:end_plot]), label='Imag (Q)', alpha=0.7, linewidth=1)
            
            # Plot magnitude envelope to visualize the pulse shape
            plt.plot(range(start_plot, end_plot), magnitude[start_plot:end_plot], 'k--', label='Magnitude', alpha=0.5)

            plt.title(f"Time Domain Pulse Check: {filename}\nSignal found at sample {peak_time_idx}")
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()
            
            # Save time-domain image into date-specific results folder as well
            try:
                parent_name_td = os.path.basename(os.path.dirname(filepath))
                import re as _re_td
                if _re_td.match(r"^\d{4}-\d{2}-\d{2}$", parent_name_td):
                    td_subdir = parent_name_td
                else:
                    td_subdir = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(filepath)))
            except Exception:
                td_subdir = time.strftime("%Y-%m-%d")

            td_out_dir = os.path.join(RESULT_DIR, td_subdir)
            if not os.path.exists(td_out_dir):
                try:
                    os.makedirs(td_out_dir)
                except OSError:
                    pass

            out_time_png = os.path.join(td_out_dir, filename.replace(".iq", "_TIME.png"))
            plt.savefig(out_time_png)
            plt.close()
            print(f"    -> Saved Time Domain image at sample {peak_time_idx} -> {out_time_png}")

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
        
        # Determine results subfolder by capture date (parent folder name), fallback to file mtime
        try:
            parent_name = os.path.basename(os.path.dirname(filepath))
            import re
            if re.match(r"^\d{4}-\d{2}-\d{2}$", parent_name):
                result_subdir = parent_name
            else:
                result_subdir = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(filepath)))
        except Exception:
            result_subdir = time.strftime("%Y-%m-%d")

        out_dir = os.path.join(RESULT_DIR, result_subdir)
        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
            except OSError:
                pass

        out_png = os.path.join(out_dir, filename.replace(".iq", ".png"))
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
    # Step 1: Scan and process any existing .iq files (recursively)
    # ==========================================
    print(f"[*] Scanning for backlog files under: {os.path.abspath(path)}")
    today = time.strftime("%Y-%m-%d")
    today_dir = os.path.join(path, today)

    all_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith('.iq'):
                all_files.append(os.path.join(root, f))

    # Separate files into non-today backlog and today's files
    non_today = [p for p in all_files if not os.path.abspath(p).startswith(os.path.abspath(today_dir) + os.sep)]
    today_files = [p for p in all_files if os.path.abspath(p).startswith(os.path.abspath(today_dir) + os.sep)]

    if non_today:
        print(f" -> Found {len(non_today)} backlog files (not in today's folder). Processing backlog...")
        non_today.sort(key=lambda p: os.path.getmtime(p))
        for fpath in non_today:
            fname = os.path.basename(fpath)
            process_and_cleanup(fpath, fname)
        print(f"[*] Backlog files cleaned up.")

        # After backlog is processed, also process today's files if present
        if today_files:
            print(f" -> Now processing {len(today_files)} files from today's folder: {today}")
            today_files.sort(key=lambda p: os.path.getmtime(p))
            for fpath in today_files:
                fname = os.path.basename(fpath)
                process_and_cleanup(fpath, fname)
            print(f"[*] Today's files processed.")
    else:
        if today_files:
            print(f" -> No backlog files. Processing {len(today_files)} files from today's folder: {today}")
            today_files.sort(key=lambda p: os.path.getmtime(p))
            for fpath in today_files:
                fname = os.path.basename(fpath)
                process_and_cleanup(fpath, fname)
            print(f"[*] Today's files processed.")
        else:
            print(f" -> No backlog files found and today's folder is empty.")

    # ==========================================
    # Step 2: Start monitoring for new .iq files
    # ==========================================
    event_handler = NewIQFileHandler()
    observer = Observer()
    
    # Register recursive directory monitoring (recursive=True to also watch date subdirectories)
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    print(f"--- WATCHDOG SERVICE STARTED ---")
    print(f"Monitoring directory: {os.path.abspath(path)}")
    print(f"Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()