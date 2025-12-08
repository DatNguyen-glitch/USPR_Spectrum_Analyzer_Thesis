from gnuradio import gr
import numpy as np
import time
import os
import threading
import queue

class AsyncFileWriter:
    """
    Async file writer that runs in a separate thread.
    Accepts data chunks via a queue and writes them without blocking the main thread.
    """
    def __init__(self):
        self.write_queue = queue.Queue(maxsize=100)  # Buffer up to 100 chunks
        self.thread = None
        self.running = False
        self.current_file = None
        self.current_filename = ""
    
    def start(self):
        """Start the writer thread."""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._writer_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the writer thread gracefully."""
        self.running = False
        # Send sentinel to unblock the queue
        try:
            self.write_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
    
    def open_file(self, filename):
        """Queue a file open command."""
        self.current_filename = filename
        self.write_queue.put(("OPEN", filename))
    
    def write_data(self, data_bytes):
        """Queue data for writing (non-blocking)."""
        try:
            self.write_queue.put_nowait(("WRITE", data_bytes))
            return True
        except queue.Full:
            print("[AsyncWriter] WARNING: Write queue full, dropping data!")
            return False
    
    def close_file(self):
        """Queue a file close command."""
        self.write_queue.put(("CLOSE", None))
    
    def _writer_loop(self):
        """Background thread loop that processes the write queue."""
        while self.running:
            try:
                item = self.write_queue.get(timeout=0.5)
                if item is None:
                    continue
                
                cmd, payload = item
                
                if cmd == "OPEN":
                    # Close any existing file first
                    if self.current_file:
                        try:
                            self.current_file.close()
                        except:
                            pass
                    # Open new file
                    try:
                        self.current_file = open(payload, "wb")
                    except Exception as e:
                        print(f"[AsyncWriter] Failed to open {payload}: {e}")
                        self.current_file = None
                
                elif cmd == "WRITE":
                    if self.current_file:
                        try:
                            self.current_file.write(payload)
                        except Exception as e:
                            print(f"[AsyncWriter] Write error: {e}")
                
                elif cmd == "CLOSE":
                    if self.current_file:
                        try:
                            self.current_file.flush()
                            self.current_file.close()
                            # Delete if empty
                            if os.path.exists(self.current_filename):
                                if os.path.getsize(self.current_filename) == 0:
                                    os.remove(self.current_filename)
                                    print(f"[AsyncWriter] Deleted empty file: {self.current_filename}")
                        except Exception as e:
                            print(f"[AsyncWriter] Close error: {e}")
                        finally:
                            self.current_file = None
                
                self.write_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AsyncWriter] Loop error: {e}")
        
        # Cleanup on exit
        if self.current_file:
            try:
                self.current_file.close()
            except:
                pass


class ring_buffer(gr.sync_block):
    """
    Ring buffer with async file I/O.
    Uses a separate thread for writing to avoid blocking the GNU Radio scheduler.
    Uses a pre-allocated NumPy circular buffer instead of deque for performance.
    """
    
    def _delete_if_empty(self, filename):
        """Delete the file if it is empty (0 bytes)."""
        try:
            if os.path.isfile(filename) and os.path.getsize(filename) == 0:
                os.remove(filename)
        except Exception:
            pass

    def __init__(self, samp_rate, buffer_ms=4, post_ms=4):
        gr.sync_block.__init__(self,
            name="ring_buffer",
            in_sig=[np.complex64],
            out_sig=[])

        self.samp_rate = samp_rate
        self.buffer_len = int(samp_rate * buffer_ms / 1000)
        self.post_len = int(samp_rate * post_ms / 1000)
        
        # Pre-allocated NumPy circular buffer (much faster than deque)
        self._ring_buf = np.zeros(self.buffer_len, dtype=np.complex64)
        self._ring_idx = 0  # Current write position
        self._ring_count = 0  # Number of valid samples (up to buffer_len)
        
        self.trigger = False
        self.center_freq = None
        self.carrier_hz = None
        self.current_filename = ""
        self.post_counter = 0
        
        # Initialize async writer
        self.async_writer = AsyncFileWriter()
        self.async_writer.start()

    def __del__(self):
        """Cleanup when block is destroyed."""
        if hasattr(self, 'async_writer'):
            self.async_writer.stop()

    def stop(self):
        """Called when flowgraph stops."""
        if hasattr(self, 'async_writer'):
            self.async_writer.stop()
        return True

    def _get_ring_buffer_contents(self):
        """Extract valid data from circular buffer in correct order."""
        if self._ring_count == 0:
            return np.array([], dtype=np.complex64)
        
        if self._ring_count < self.buffer_len:
            # Buffer not yet full, data starts at 0
            return self._ring_buf[:self._ring_count].copy()
        else:
            # Buffer is full, need to unwrap from write position
            return np.concatenate((
                self._ring_buf[self._ring_idx:],
                self._ring_buf[:self._ring_idx]
            ))
    
    def _convert_to_int16_fast(self, data):
        """Convert complex64 to int16 (I/Q interleaved)."""
        if len(data) == 0:
            return b''
        float_view = data.view(np.float32)
        scaled = float_view * (32767.0 * 4.0)
        clipped = np.clip(scaled, -32768, 32767)
        int16_data = clipped.astype(np.int16)
        return int16_data.tobytes()

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
            # Open file via async writer
            self.async_writer.open_file(self.current_filename)
            
            # Write pre-trigger buffer (async) - use optimized extraction
            if self._ring_count > 0:
                arr = self._get_ring_buffer_contents()
                self.async_writer.write_data(self._convert_to_int16_fast(arr))
                
        except Exception as e:
            print(f"[RingBuffer] Trigger Error: {e}")
            self.trigger = False
            self.current_filename = ""

    def work(self, input_items, output_items):
        data = input_items[0]
        n_data = len(data)
        
        # Update circular buffer efficiently using numpy operations
        # This is O(n) but with numpy's optimized memory operations
        if n_data >= self.buffer_len:
            # Data is larger than buffer, just take the last buffer_len samples
            self._ring_buf[:] = data[-self.buffer_len:]
            self._ring_idx = 0
            self._ring_count = self.buffer_len
        else:
            # Calculate how much fits before wrap-around
            space_to_end = self.buffer_len - self._ring_idx
            
            if n_data <= space_to_end:
                # All data fits without wrapping
                self._ring_buf[self._ring_idx:self._ring_idx + n_data] = data
                self._ring_idx = (self._ring_idx + n_data) % self.buffer_len
            else:
                # Need to wrap around
                self._ring_buf[self._ring_idx:] = data[:space_to_end]
                remaining = n_data - space_to_end
                self._ring_buf[:remaining] = data[space_to_end:]
                self._ring_idx = remaining
            
            self._ring_count = min(self._ring_count + n_data, self.buffer_len)

        if self.trigger:
            try:
                n_write = min(n_data, self.post_counter)
                
                if n_write > 0:
                    chunk = data[:n_write]
                    # Queue for async write (non-blocking!)
                    self.async_writer.write_data(self._convert_to_int16_fast(chunk))
                    self.post_counter -= n_write

                if self.post_counter <= 0:
                    # Queue file close (async)
                    self.async_writer.close_file()
                    self.trigger = False
                    
            except Exception as e:
                print(f"[RingBuffer] Work Error: {e}")
                self.trigger = False
                self.async_writer.close_file()
                
        return n_data