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

    chan_count = 61
    channelizer = PolyphaseChannelizer(chan_count)
    chan_width = fs / chan_count

    print("Channelizing")
    t0 = time()
    channelized = channelizer.process(samples)
    t1 = time()
    print("Channelized %.3f s of samples in %.3f s" % (len(samples) / fs, t1 - t0))
    return

    print("Filtering")
    INIT_DECIM = 4
    fs = sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, channel) // INIT_DECIM
    lpf = scipy.signal.butter(3, 1E6, fs=fs)
    capf = scipy.signal.lfilter(*lpf, captures[0][::INIT_DECIM])
    ds = capf[::2]
    fs //= 2

    print("Extract bursts")
    bursts = burst_extract(ds)

    print("Demod")
    samps_per_sym = fs / 1E6
    for b in bursts:
        syms = fsk_decode(b, samps_per_sym, True)
        offset = find_sync32(syms)
        if offset:
            data = unpack_syms(syms, offset)
            data_dw = le_dewhiten(data[4:], 37)
            pkt = le_trim_pkt(data_dw)
            print(hex_str(pkt))
        else:
            print("sync not found")

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
