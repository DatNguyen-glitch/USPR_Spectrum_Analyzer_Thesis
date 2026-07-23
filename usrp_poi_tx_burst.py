#!/usr/bin/env python3
"""
TX burst scheduler cho thí nghiệm POI (Probability of Intercept).

Phát các burst CW tại MỘT tần số cố định f_burst. Mỗi burst dài đúng b giây,
căn thời gian bằng UHD timed-command (start_of_burst / end_of_burst + time_spec)
nên độ chính xác ở mức MẪU: b = 2 ms là 2 ms thật. Giữa các burst DAC nghỉ hẳn
(khoảng lặng thật, không phát zero) — đúng mô hình burst rời rạc mà POI cần.

Ghi ground-truth (burst_id, b, t_on, t_off) theo wall-clock để đối chiếu offline
với FFT frame ghi ở máy RX. Vì 2 máy đồng bộ PTP (ptp4l + phc2sys, CLOCK_REALTIME
khớp sub-µs), timestamp scheduled ở đây so sánh trực tiếp được với "ts" của frame RX.

QUAN TRỌNG: chạy MỘT giá trị b cho mỗi run (--burst-ms). N >= 1000 burst/run.

Vì sao dùng UHD trực tiếp thay vì GNU Radio: timed burst có gap là tính năng
UHD-native; GNU Radio thiết kế cho stream liên tục nên phát burst thưa rất khó.
"""

import argparse
import csv
import json
import os
import random
import signal
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np
import uhd


# ── Mặc định (chỉnh qua CLI) ─────────────────────────────────────────────────
DEF_SERIAL   = "34D62A7"     # serial của USRP TX
DEF_FBURST   = 436e6         # tần số RF của tone (rơi vào hop 433, lệch tâm -7 MHz)
DEF_IF       = 1e6           # dịch IF: tone ở baseband +1 MHz để né LO leakage tại DC
DEF_FS       = 5e6           # sample rate TX (thừa cho tone 1 MHz)
DEF_GAIN     = 15
DEF_BW       = 5e6           # analog LPF
DEF_AMP      = 0.3           # biên độ số, dưới 1.0 để không tràn DAC
DEF_NBURST   = 1000
DEF_MEANGAP  = 0.5           # trung bình khoảng nghỉ (Poisson) — >> T_sweep ~0.133 s
MIN_GAP      = 0.2           # sàn khoảng nghỉ, đảm bảo 2 burst không dồn vào 1 lần quét
LEAD         = 0.1           # đẩy mẫu tới UHD trước thời điểm phát 100 ms (tránh lỗi "late")
FIRST_DELAY  = 0.5           # burst đầu cách thời điểm start 0.5 s

stop_event = threading.Event()


def make_tone_2d(n_samples: int, if_hz: float, fs: float, amplitude: float) -> np.ndarray:
    """Sinh 1 burst tone phức e^(j2π·if·t), trả về mảng 2D (1, N) cho send()."""
    n = np.arange(n_samples, dtype=np.float64)
    wf = amplitude * np.exp(1j * 2.0 * np.pi * (if_hz / fs) * n)
    return np.ascontiguousarray(wf.astype(np.complex64)).reshape(1, -1)


def send_burst(tx_streamer, wf2d: np.ndarray, t_sched_s: float, timeout: float = 1.0) -> int:
    """Phát 1 burst căn giờ tại t_sched_s (giây, theo đồng hồ USRP ≈ wall-clock).

    time_spec + start_of_burst gắn vào chunk ĐẦU; end_of_burst gắn vào chunk CUỐI.
    Nếu burst lớn hơn buffer thì gửi nhiều chunk, chỉ chunk đầu mang time_spec/sob.
    """
    md = uhd.types.TXMetadata()
    md.has_time_spec = True
    md.time_spec = uhd.types.TimeSpec(float(t_sched_s))
    md.start_of_burst = True
    md.end_of_burst = False

    total = wf2d.shape[1]
    max_samps = tx_streamer.get_max_num_samps()
    sent = 0
    while sent < total:
        n = min(max_samps, total - sent)
        md.end_of_burst = (sent + n >= total)
        nsent = tx_streamer.send(wf2d[:, sent:sent + n], md, timeout)
        # sau chunk đầu: không lặp lại time_spec / sob
        md.has_time_spec = False
        md.start_of_burst = False
        if nsent == 0:            # timeout khi đẩy mẫu → bỏ phần còn lại của burst
            break
        sent += nsent
    return sent

def async_monitor(tx_streamer, err_path: str):
    """Luồng phụ giám sát lỗi phần cứng thời gian thực từ USRP.
    Sử dụng mốc thời gian cứng của FPGA và so sánh Enum trực tiếp để né bẫy GIL/Trễ mạng.
    """
    try:
    	EventCode = uhd.types.TXMetadataEventCode
    	amd = uhd.types.TXAsyncMetadata()
    except Exception as e:
    	print(f"[async] monitor OFF (UHD API mismatch): {e}", file=sys.stderr)
    	return

    with open(err_path, "a", encoding="utf-8") as f:
        f.write("# usrp_time,event_code,host_time\n")
        f.flush()
        
        while not stop_event.is_set():
            try:
                # Đọc bản tin với timeout ngắn (0.05s) để luồng có thể check stop_event liên tục
                if tx_streamer.recv_async_msg(amd, 0.05):
                    # 1. Bỏ qua bản tin ACK thành công một cách nhanh nhất bằng so sánh Enum
                    if amd.event_code == EventCode.burst_ack:
                        continue
                    
                    # 2. Lấy mốc thời gian thực từ phần cứng USRP (nếu có)
                    if amd.has_time_spec:
                        event_time = f"{amd.time_spec.get_real_secs():.9f}"
                    else:
                        event_time = f"NOT_AVAILABLE" # Fallback nếu phần cứng không hỗ trợ tag thời gian cho lỗi đó

                    event_name = str(amd.event_code)
                    
                    # 3. Ghi log: lưu cả thời gian USRP (để map với TX) và thời gian Host (để debug hệ thống)
                    f.write(f"{event_time},{event_name},{time.time():.6f}\n")
                    f.flush()
                    
                    print(f"[async][WARN] Phát hiện lỗi phần cứng: {event_name} tại USRP Time: {event_time}", file=sys.stderr)
            
            except Exception as e:
                # Tránh loop-spam phá hủy CPU nếu API driver bị lỗi cục bộ lúc tear-down
                time.sleep(0.05)

def parse_args():
    p = argparse.ArgumentParser(description="POI burst TX (UHD timed-command)")
    p.add_argument("--burst-ms", type=float, default=5.0,
                   help="độ dài burst (ms). Chạy 1 giá trị/run: {2, 5, 10, 20}")
    p.add_argument("--n-bursts", type=int, default=DEF_NBURST)
    p.add_argument("--mean-gap", type=float, default=DEF_MEANGAP,
                   help="trung bình khoảng nghỉ Poisson (s); phải >> T_sweep")
    p.add_argument("--freq", type=float, default=DEF_FBURST, help="f_burst RF (Hz)")
    p.add_argument("--if-offset", type=float, default=DEF_IF)
    p.add_argument("--fs", type=float, default=DEF_FS)
    p.add_argument("--gain", type=float, default=DEF_GAIN)
    p.add_argument("--bw", type=float, default=DEF_BW)
    p.add_argument("--amp", type=float, default=DEF_AMP)
    p.add_argument("--serial", type=str, default=DEF_SERIAL)
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--no-monitor", action="store_true", help="tắt async error monitor")
    return p.parse_args()


def main():
    args = parse_args()

    if args.amp > 1.0:
        print("[WARN] amp > 1.0 sẽ tràn DAC", file=sys.stderr)
    if args.gain > 60:
        print("[WARN] gain cao — kiểm tra RX không bị bão hòa ADC", file=sys.stderr)

    b_sec = args.burst_ms / 1000.0
    lo_freq = args.freq - args.if_offset       # LO lùi lại; tone rơi đúng vào args.freq

    signal.signal(signal.SIGINT,  lambda s, f: stop_event.set())
    signal.signal(signal.SIGTERM, lambda s, f: stop_event.set())

    # ── Khởi tạo USRP ────────────────────────────────────────────────────────
    usrp = uhd.usrp.MultiUSRP(f"serial={args.serial}")
    usrp.set_tx_rate(args.fs)
    usrp.set_tx_freq(uhd.types.TuneRequest(lo_freq), 0)
    usrp.set_tx_gain(args.gain, 0)
    usrp.set_tx_antenna("TX/RX", 0)
    usrp.set_tx_bandwidth(args.bw, 0)

    # đọc lại giá trị THỰC (rate/freq có thể bị coerce) để tính đúng tone & bin
    actual_fs   = float(usrp.get_tx_rate())
    actual_cen  = float(usrp.get_tx_freq(0))
    actual_tone = actual_cen + args.if_offset          # tần số RF thật của tone

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [0]
    tx_streamer = usrp.get_tx_stream(st_args)

    # đồng bộ đồng hồ USRP về wall-clock (đã PTP-sync). δ ~sub-ms không ảnh hưởng:
    # intercept dựa vào NĂNG LƯỢNG FFT, timestamp chỉ để khoanh vùng frame ứng viên.
    usrp.set_time_now(uhd.types.TimeSpec(time.time()))

    # 1 burst = round(b × fs_thực) mẫu → dùng fs THỰC để tone đúng if_offset
    n_per_burst = int(round(b_sec * actual_fs))
    waveform = make_tone_2d(n_per_burst, args.if_offset, actual_fs, args.amp)

    # ── File output ──────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(args.out_dir, f"poi_tx_b{args.burst_ms:g}ms_{stamp}")
    csv_path, meta_path, err_path = base + ".csv", base + ".meta.json", base + ".err.csv"

    t_start = usrp.get_time_now().get_real_secs() + FIRST_DELAY

    meta = {
        "experiment": "POI_burst_tx",
        "f_burst_hz_requested": args.freq,
        "f_burst_hz_actual": actual_tone,
        "lo_hz_actual": actual_cen,
        "if_offset_hz": args.if_offset,
        "fs_hz_actual": actual_fs,
        "tx_gain_db": args.gain,
        "tx_bw_hz": args.bw,
        "amplitude": args.amp,
        "burst_ms": args.burst_ms,
        "samples_per_burst": n_per_burst,
        "n_bursts": args.n_bursts,
        "mean_gap_s": args.mean_gap,
        "min_gap_s": MIN_GAP,
        "start_wall_time": t_start,
        "start_iso_utc": datetime.now(timezone.utc).isoformat(),
        "sync": "ptp4l+phc2sys, CLOCK_REALTIME sub-us; USRP time = wall-clock",
        "serial": args.serial,
        "note": "t_on/t_off theo đồng hồ USRP ≈ wall-clock; so trực tiếp với ts frame RX",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"== POI TX burst ==")
    print(f"  f_burst   : {actual_tone/1e6:.6f} MHz (LO {actual_cen/1e6:.6f} + IF {args.if_offset/1e6:g})")
    print(f"  burst     : {args.burst_ms:g} ms = {n_per_burst} mẫu @ {actual_fs/1e6:g} Msps")
    print(f"  n_bursts  : {args.n_bursts}, mean_gap {args.mean_gap}s (sàn {MIN_GAP}s)")
    print(f"  ước tính  : ~{args.n_bursts*(b_sec+args.mean_gap)/60:.1f} phút")
    print(f"  ground-truth -> {csv_path}")

    # ── Async monitor (tùy chọn) ─────────────────────────────────────────────
    mon_thread = None
    if not args.no_monitor:
        mon_thread = threading.Thread(target=async_monitor, args=(tx_streamer, err_path), daemon=True)
        mon_thread.start()

    # ── Vòng phát burst ──────────────────────────────────────────────────────
    sent_ok = 0
    t_next = t_start
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["burst_id", "b_ms", "t_on", "t_off"])
        try:
            for burst_id in range(args.n_bursts):
                if stop_event.is_set():
                    break

                # pace: ngủ tới LEAD giây trước thời điểm phát (dùng đồng hồ USRP)
                wait = (t_next - LEAD) - usrp.get_time_now().get_real_secs()
                if wait > 0:
                    time.sleep(wait)

                nsent = send_burst(tx_streamer, waveform, t_next)
                if nsent == n_per_burst:
                    sent_ok += 1

                # ghi ground-truth: thời điểm PHÁT theo lịch (= thời điểm RF thật, sai số sub-ms)
                w.writerow([burst_id, f"{args.burst_ms:g}", f"{t_next:.9f}", f"{t_next + b_sec:.9f}"])

                if burst_id % 50 == 0:
                    cf.flush()
                    print(f"  [{burst_id}/{args.n_bursts}] t_on={t_next:.6f} sent={nsent}")

                # lịch burst kế: khoảng nghỉ Poisson (exponential), sàn MIN_GAP
                gap = MIN_GAP + random.expovariate(1.0 / args.mean_gap)
                t_next = t_next + b_sec + gap
        finally:
            cf.flush()

    # đợi burst cuối thực sự phát xong trước khi tear-down
    time.sleep(b_sec + LEAD + 0.3)
    stop_event.set()
    if mon_thread is not None:
        mon_thread.join(timeout=1.0)

    print(f"== xong: {sent_ok}/{args.n_bursts} burst phát đủ mẫu ==")
    print(f"   CSV : {csv_path}")
    print(f"   meta: {meta_path}")


if __name__ == "__main__":
    main()
