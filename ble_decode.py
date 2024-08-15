#!/usr/bin/env python3

import SoapySDR
import numpy
import scipy.signal
from time import time
import matplotlib.pyplot as plt
from struct import pack, unpack

from channelizer import PolyphaseChannelizer
from ble_utils import *

def main():
    print("Loading data")
    fs = 122.88e6
    samples = numpy.fromfile("ble_capture_f_2440_sr_122880000.cf32", dtype=numpy.complex64)

    print("Resampling")
    # 122.88 -> 96:  25/32
    RESAMP_UP = 25
    RESAMP_DOWN = 32
    fs = fs * RESAMP_UP / RESAMP_DOWN
    filt_size = 5 * RESAMP_UP
    h = scipy.signal.firwin(filt_size, 0.9 / RESAMP_DOWN).astype(numpy.complex64) * RESAMP_UP
    t0 = time()
    samples = scipy.signal.upfirdn(h, samples, RESAMP_UP, RESAMP_DOWN)
    t1 = time()
    print("resampled in", t1 - t0, "s")

    chan_count = 48
    channelizer = PolyphaseChannelizer(chan_count)
    chan_width = fs / chan_count

    channels_ble = [37, 38, 39]
    channels_seq = [0, 12, 39]
    centre_seq = 19 # 2440 MHz
    channels_poly = [channelizer.chan_idx(s - centre_seq) for s in channels_seq]

    print("Channelizing and processing")
    t0 = time()
    # chunk size of 2^22 tuned for performance on 6-core M2 Pro with Mac OS 14
    # smaller chunk sizes get worse performance, below 2^20 is dramatically worse
    # bigger chunk sizes also actually get a little worse on my Mac
    chunk_sz = 1 << 22
    for i in range(0, len(samples), chunk_sz):
        channelized = channelizer.process(samples[i:i + chunk_sz])
        process_channels(channelized, chan_width, channels_ble, channels_poly)
    t1 = time()
    print("Processed %.3f s of samples in %.3f s" % (len(samples) / fs, t1 - t0))
    print("Found %d, failed %d" % (found, failed))

found = 0
failed = 0

def process_channels(channelized, fs, channels_ble, channels_poly):
    global found, failed

    for i, chan in enumerate(channels_ble):
        samples = channelized[channels_poly[i]]
        samples_demod = fm_demod(samples) > 0
        peaks = find_sync_multi2(samples_demod, b'\xd6\xbe\x89\x8e')
        pkts = ble_pkt_extract(samples_demod, peaks, chan)
        for pkt in pkts:
            print(chan, hex_str(pkt))
            found += 1

    """
    print("Plotting")
    fig, axs = plt.subplots(3)
    axs[0].plot(numpy.real(ds))
    axs[1].plot(numpy.real(squelch(ds)))
    axs[2].plot(fm_demod(squelch(ds)))
    fig.tight_layout()
    plt.show()
    """

if __name__ == "__main__":
    main()
