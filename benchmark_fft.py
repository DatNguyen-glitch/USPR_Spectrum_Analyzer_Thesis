import time
import sys
import numpy as np
from gnuradio import gr, blocks, fft
from gnuradio.fft import window

class FFTBenchmarkFC32(gr.top_block):
    def __init__(self, fft_size, num_vectors):
        gr.top_block.__init__(self, "FFT Benchmark FC32")

        # ---------------------------------------------------------
        # IMPORTANT: Generate synthetic data in the correct "fc32" format
        # fc32 in UHD = np.complex64 in Python (2 x 32-bit float)
        # ---------------------------------------------------------
        total_samples = fft_size * num_vectors
        
        # 1. Create random real/imag parts (default is float64)
        real_part = np.random.randn(total_samples)
        imag_part = np.random.randn(total_samples)
        
        # 2. Cast to complex64 to match cpu_format="fc32"
        # This ensures memory layout and bandwidth match the USRP signal.
        src_data = (real_part + 1j * imag_part).astype(np.complex64)
        
        # Quick sanity-check the item size (must be 8 bytes)
        if src_data.itemsize != 8:
            print(f"[ERROR] Wrong format! Item size is {src_data.itemsize} bytes; expected 8 bytes (fc32).")
            sys.exit(1)

        # Source: Vector Source outputs fc32 data
        self.src = blocks.vector_source_c(src_data, False)
        
        # Stream to Vector: Pack samples into vectors for the FFT
        self.s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, fft_size)
        
        # FFT Block: This is the main object being benchmarked
        # Use the Blackman-Harris window identical to your main code
        my_window = window.blackmanharris(fft_size)
        self.fft = fft.fft_vcc(fft_size, True, my_window, True, 1)
        
        # Sink: Null Sink (discard data to avoid disk bottlenecks)
        self.sink = blocks.null_sink(gr.sizeof_gr_complex * fft_size)

        # Connect flowgraph
        self.connect(self.src, self.s2v)
        self.connect(self.s2v, self.fft)
        self.connect(self.fft, self.sink)

def run_test(fft_size, target_sps=50e6):
    # Simulate processing ~50 million samples (~1 second at 50 MSps)
    total_samples = 50000000 
    # Round down number of vectors
    num_vectors = int(total_samples // fft_size)
    
    # Initialize flowgraph
    tb = FFTBenchmarkFC32(fft_size, num_vectors)
    
    # Start timing
    start_time = time.time()
    tb.run() # Run blocking until all data has been processed
    end_time = time.time()
    
    elapsed = end_time - start_time
    if elapsed < 1e-6: elapsed = 1e-6 # Avoid division by zero
    
    # Compute metrics
    samples_processed = num_vectors * fft_size
    throughput = samples_processed / elapsed
    
    print(f"[-] FFT Size: {fft_size:4d} | Samples: {samples_processed} (fc32) | Time: {elapsed:.4f}s")
    print(f"    -> Actual throughput: {throughput/1e6:.2f} MSps")
    
    # Check whether we meet the target rate
    if throughput > target_sps:
        margin = ((throughput - target_sps) / target_sps) * 100
        print(f"    => CONCLUSION: \033[92mPASS\033[0m (Margin {margin:.1f}% vs 50 MSps)")
    else:
        missing = ((target_sps - throughput) / target_sps) * 100
        print(f"    => CONCLUSION: \033[91mFAIL\033[0m (Short by {missing:.1f}%)")
    print("-" * 60)

if __name__ == "__main__":
    print("=== BENCHMARK FFT (Format: fc32 / complex64) ===")
    print("System target: 50 MSps (Sample Rate = 50M)")
    print("-" * 60)
    
    # FFT sizes under consideration
    test_sizes = [512, 1024, 2048, 4096, 8192]

    for n in test_sizes:
        run_test(n)