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
#include <chrono>

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
        d_consec_count(0),
        d_skip_vectors(0)
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
        if (d_temp_vec.size() != (size_t)d_vec_len) {
            d_temp_vec.resize((size_t)d_vec_len);
        }

        // Get tag to update center frequency if available
        std::vector<tag_t> tags;
        get_tags_in_range(tags, 0, nitems_read(0), nitems_read(0) + noutput_items);
        
        std::sort(tags.begin(), tags.end(), [](const tag_t &a, const tag_t &b) {
            return a.offset < b.offset;
        });

        // Iterator for tags
        auto tag_it = tags.begin();

        // for (auto &t : tags) {
        //     if (pmt::eq(t.key, pmt::mp("rx_freq"))) {
        //         d_center_freq = (float)pmt::to_float(t.value);
        //         GR_LOG_INFO(d_logger, boost::format("Updated center frequency to %.2f MHz") % (d_center_freq/1e6));
        //         // Reset counter when freq changes
        //         d_consec_count = 0; 
        //     }
        // }

        // Iterate over each input vector
        for (int i = 0; i < noutput_items; i++) {
            uint64_t current_abs_sample = nitems_read(0) + i;
            while (tag_it != tags.end() && tag_it->offset <= current_abs_sample) {
            if (pmt::eq(tag_it->key, pmt::mp("rx_freq"))) {
                d_center_freq = (float)pmt::to_float(tag_it->value);
                int vectors_to_skip = (int)((d_samp_rate * 0.001f) / d_vec_len);
                d_skip_vectors = std::max(5, vectors_to_skip); // Ít nhất là 5 frame
                auto now = std::chrono::system_clock::now();
                auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    now.time_since_epoch()).count();
                // std::cout << "Time " << milliseconds << " Freq changed to " << d_center_freq << ". Skipping " << d_skip_vectors << " frames." << std::endl;
            }
            tag_it++;
            }
            // Logic Blanking
            if (d_skip_vectors > 0) {
                d_skip_vectors--;
                continue;
            }

            const float *vec_start = &in[i * d_vec_len];

            // Ignore/mask DC bins around center before noise-floor estimation and clustering
            const int dc_ignore_width = 4; // bins on each side of DC (inclusive)
            const int dc = d_vec_len / 2;
            
            // Prepare noise-only vector (Mask DC bins)
            int noise_len = 0;
            for (int k = 0; k < d_vec_len; k++) {
                if (std::abs(k - dc) <= dc_ignore_width) {
                    continue;
                }
                d_temp_vec[noise_len++] = vec_start[k];
            }

            // Calculate Noise Floor (Median Estimation) + Threshold
            float noise_floor = 0.0f;
            if (noise_len > 0) {
                std::nth_element(d_temp_vec.begin(),
                                 d_temp_vec.begin() + noise_len / 2,
                                 d_temp_vec.begin() + noise_len);
                noise_floor = d_temp_vec[noise_len / 2];
            } else {
                // Fallback (shouldn't happen for typical vec_len)
                noise_floor = vec_start[dc];
            }
            float threshold = noise_floor + d_margin_db;
            // if (i == 0 && (nitems_read(0) % 1000 == 0)) {
            //     auto now = std::chrono::system_clock::now();
            //     auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
            //                         now.time_since_epoch()).count();
            //     std::cout << "[DEBUG] Time " << milliseconds<< " Noise: " << noise_floor << " | Thresh: " << threshold << std::endl;
            // }

            // Find Clusters (Single Pass Loop - O(N))
            bool in_cluster = false;
            int cluster_start = 0;
            
            Cluster best_cluster = {0, 0, 0, -9999.0f};
            bool found_any = false;
            for (int k = 0; k < d_vec_len; k++) {
                // If in DC region, treat as not high
                bool is_dc_region = (std::abs(k - dc) <= dc_ignore_width);
                bool is_high = (!is_dc_region) && (vec_start[k] > threshold);

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

            if (d_consec_count >= 1 && found_any) { // persistence_k = 1
                float carrier_hz = compute_freq_for_bin(best_cluster.peak_idx);
                float bw_hz = (best_cluster.end - best_cluster.start + 1) * d_df;
                float snr = best_cluster.peak_power - noise_floor;
                auto now = std::chrono::system_clock::now();
                auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    now.time_since_epoch()).count();
                std::cout << "DEBUG C++: Time= " << milliseconds 
                        << " Peak=" << best_cluster.peak_power 
                        << " Noise=" << noise_floor 
                        << " Thresh=" << threshold << std::endl;
                // print log
                GR_LOG_INFO(d_logger, boost::format("Detected:Center_Freq=%.2f, Freq=%.4f Hz, BW=%.2f Hz, SNR=%.2f dB") % (d_center_freq/1e6) % (carrier_hz/1e6) % bw_hz % snr);
                // GR_LOG_INFO(d_logger, boost::format("Detected:Freq=%.4f Hz, BW=%.2f Hz, SNR=%.2f dB") % (carrier_hz/1e6) % bw_hz % snr);
                // std::string json_str = "{\"freq\": " + std::to_string(carrier_hz) +     // 4. Tạo PDU Message (JSON format)
                //                        ", \"bw\": " + std::to_string(bw_hz) + 
                //                        ", \"snr\": " + std::to_string(snr) + "}";

                // // Payload (Byte vector)
                // std::vector<uint8_t> data_vec(json_str.begin(), json_str.end());
                // pmt::pmt_t payload = pmt::init_u8vector(data_vec.size(), data_vec);
                
                // // Metadata
                // pmt::pmt_t meta = pmt::make_dict();
                // pmt::pmt_t tx_sob = pmt::intern("tx_sob");
                // pmt::pmt_t tx_eob = pmt::intern("tx_eob");
                // pmt::dict_add(meta, tx_sob, pmt::PMT_T);
                // pmt::dict_add(meta, tx_eob, pmt::PMT_T);

                // // Publish
                // message_port_pub(d_port_id, pmt::cons(meta, payload));
                
                // Reset count
                d_consec_count = 0;
            }
        }

        // Notify the scheduler that processing is done
        return noutput_items;
    }
} /* namespace specdetect */
} /* namespace gr */
