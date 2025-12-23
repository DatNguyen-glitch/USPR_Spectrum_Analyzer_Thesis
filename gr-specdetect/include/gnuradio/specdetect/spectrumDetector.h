/* -*- c++ -*- */
/*
 * Copyright 2025 NguyenTienDat.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_SPECDETECT_SPECTRUMDETECTOR_H
#define INCLUDED_SPECDETECT_SPECTRUMDETECTOR_H

#include <gnuradio/specdetect/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace specdetect {

/*!
 * \brief <+description of block+>
 * \ingroup specdetect
 *
 */
class SPECDETECT_API spectrumDetector : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<spectrumDetector> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of specdetect::spectrumDetector.
     *
     * To avoid accidental use of raw pointers, specdetect::spectrumDetector's
     * constructor is in a private implementation
     * class. specdetect::spectrumDetector::make is the public interface for
     * creating new instances.
     */
    static sptr make(int vec_len, float samp_rate, float margin_db, float min_bw_hz);
    virtual void set_center_freq(float freq) = 0;
};

} // namespace specdetect
} // namespace gr

#endif /* INCLUDED_SPECDETECT_SPECTRUMDETECTOR_H */
