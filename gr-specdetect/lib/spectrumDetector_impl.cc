/* -*- c++ -*- */
/*
 * Copyright 2025 NguyenTienDat.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "spectrumDetector_impl.h"
#include <gnuradio/io_signature.h>
#include <gnuradio/logger.h>
#include <algorithm> // std::max_element
#include <vector>
#include <cmath>
#include <string>

namespace gr {
namespace specdetect {
    void spectrumDetector_impl::set_center_freq(float freq) {
        d_center_freq = freq;
    }

    float spectrumDetector_impl::compute_freq_for_bin(int k) {
        float f_start = d_center_freq - (d_samp_rate / 2.0f);
        // return d_center_freq - (d_samp_rate / 2.0f) + (k * d_df);
        return f_start + (k * d_df);
    }

spectrumDetector::sptr
spectrumDetector::make(int vec_len, float samp_rate, float margin_db, float min_bw_hz)
{
    return gnuradio::make_block_sptr<spectrumDetector_impl>(
        vec_len, samp_rate, margin_db, min_bw_hz);
}


/*
 * The private constructor
 */
spectrumDetector_impl::spectrumDetector_impl(int vec_len, float samp_rate, float margin_db, float min_bw_hz)
    : gr::sync_block("spectrumDetector",
                     gr::io_signature::make(
                         1, 1, sizeof(float) * vec_len),    // Input: Vector float
                     gr::io_signature::make(
                         0, 0, 0)),                         // Output: No stream out
        d_vec_len(vec_len),
        d_samp_rate(samp_rate),
        d_center_freq(0.0f),
        d_margin_db(margin_db),
        d_min_bw_hz(min_bw_hz),
        d_consec_count(0)
{
    d_df = d_samp_rate / (float)d_vec_len;
    d_min_bins = std::max(1, (int)std::ceil(d_min_bw_hz / d_df));
    d_temp_vec.resize(d_vec_len);
    
    // message output port
    d_port_id = pmt::mp("msg_out");
    message_port_register_out(d_port_id);
}
/*
 * Our virtual destructor.
 */
spectrumDetector_impl::~spectrumDetector_impl() {}

int spectrumDetector_impl::work(int noutput_items,
                                   gr_vector_const_void_star& input_items,
                                   gr_vector_void_star& output_items)
    {
        const float *in = (const float *)input_items[0];

        // Get tag to update center frequency if available
        std::vector<tag_t> tags;
        get_tags_in_range(tags, 0, nitems_read(0), nitems_read(0) + noutput_items);
        
        for (auto &t : tags) {
            if (pmt::eq(t.key, pmt::mp("rx_freq"))) {
                d_center_freq = (float)pmt::to_float(t.value);
                // Reset bộ đếm khi đổi tần số
                d_consec_count = 0; 
            }
        }

        // Iterate over each input vector
        for (int i = 0; i < noutput_items; i++) {
            const float *vec_start = &in[i * d_vec_len];
            
            // 1. Calculate Noise Floor (Median Estimation)
            // For speed, we can use mean or sample a few points instead of sorting the entire vector
            // Quick way: Calculate the mean of the lowest 10% samples (simulate noise)
            // Here we use a simple way: Take the mean (if noise is flat) or Min + offset
            // To be as accurate as Python: Copy to vector, sort and take median (still much faster than Python)
            std::copy(vec_start, vec_start + d_vec_len, d_temp_vec.begin());

            std::nth_element(d_temp_vec.begin(), d_temp_vec.begin() + d_vec_len/2, d_temp_vec.end());
            float noise_floor = d_temp_vec[d_vec_len/2];
            
            float threshold = noise_floor + d_margin_db;

            // 2. Find Clusters (Single Pass Loop - O(N))
            bool in_cluster = false;
            int cluster_start = 0;
            
            Cluster best_cluster = {0, 0, 0, -9999.0f};
            bool found_any = false;
            const int dc_ignore_width = 4; // Ignore bins around DC
            for (int k = 0; k < d_vec_len; k++) {
                if (k <= dc_ignore_width || k >= (d_vec_len - dc_ignore_width)) {
                    continue; // Ignore DC bins
                }
                bool is_high = vec_start[k] > threshold;

                if (is_high && !in_cluster) {
                    // Start rising edge
                    in_cluster = true;
                    cluster_start = k;
                } else if (!is_high && in_cluster) {
                    // End falling edge
                    in_cluster = false;
                    int cluster_end = k - 1;
                    int width = cluster_end - cluster_start + 1;

                    if (width >= d_min_bins) {
                        // Find peak within this cluster
                        auto max_it = std::max_element(
                        vec_start + cluster_start,
                        vec_start + cluster_end + 1);

                        float peak_val = *max_it;
                        int peak_idx = max_it - vec_start;

                        if (peak_val > best_cluster.peak_power) {
                            best_cluster.start = cluster_start;
                            best_cluster.end = cluster_end;
                            best_cluster.peak_idx = peak_idx;
                            best_cluster.peak_power = peak_val;
                        }

                        found_any = true;
                    }
                }
            }
            // Handle case if still in cluster at end of vector
            if (in_cluster) {
                int cluster_end = d_vec_len - 1;
                int width = cluster_end - cluster_start + 1;
                if (width >= d_min_bins) {
                    auto max_it = std::max_element(
                    vec_start + cluster_start,
                    vec_start + cluster_end + 1);

                    float peak_val = *max_it;
                    int peak_idx = max_it - vec_start;

                    if (peak_val > best_cluster.peak_power) {
                        best_cluster.start = cluster_start;
                        best_cluster.end = cluster_end;
                        best_cluster.peak_idx = peak_idx;
                        best_cluster.peak_power = peak_val;
                    }

                    found_any = true;
                }
            }

            // 3. Logic Persistence & Reporting
            if (found_any) {
                d_consec_count++;
            } else {
                d_consec_count = 0;
            }

            if (d_consec_count >= 2 && found_any) { // persistence_k = 2
                float carrier_hz = compute_freq_for_bin(best_cluster.peak_idx);
                float bw_hz = (best_cluster.end - best_cluster.start + 1) * d_df;
                float snr = best_cluster.peak_power - noise_floor;

                // print log
                GR_LOG_INFO(d_logger, boost::format("Detected:Center_Freq=%.2f, Freq=%.4f Hz, BW=%.2f Hz, SNR=%.2f dB") % (d_center_freq/1e6) % (carrier_hz/1e6) % bw_hz % snr);
                // GR_LOG_INFO(d_logger, boost::format("Detected:Freq=%.4f Hz, BW=%.2f Hz, SNR=%.2f dB") % (carrier_hz/1e6) % bw_hz % snr);
                std::string json_str = "{\"freq\": " + std::to_string(carrier_hz) +     // 4. Tạo PDU Message (JSON format)
                                       ", \"bw\": " + std::to_string(bw_hz) + 
                                       ", \"snr\": " + std::to_string(snr) + "}";

                // Payload (Byte vector)
                std::vector<uint8_t> data_vec(json_str.begin(), json_str.end());
                pmt::pmt_t payload = pmt::init_u8vector(data_vec.size(), data_vec);
                
                // Metadata
                pmt::pmt_t meta = pmt::make_dict();
                pmt::pmt_t tx_sob = pmt::intern("tx_sob");
                pmt::pmt_t tx_eob = pmt::intern("tx_eob");
                pmt::dict_add(meta, tx_sob, pmt::PMT_T);
                pmt::dict_add(meta, tx_eob, pmt::PMT_T);

                // Publish
                message_port_pub(d_port_id, pmt::cons(meta, payload));
                
                // Reset count
                d_consec_count = 0;
            }
        }

        // Notify the scheduler that processing is done
        return noutput_items;
    }
} /* namespace specdetect */
} /* namespace gr */
