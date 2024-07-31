#!/usr/bin/env python3

import SoapySDR
import numpy
import scipy.signal
from time import time
import matplotlib.pyplot as plt
from struct import pack, unpack
from ble_utils import *

NUM_CHANNELS = 1
CHUNK_SZ = 1 << 18

def main():
    print("Opening RFNM")
    args = dict(driver="rfnm")
    sdr = SoapySDR.Device(args)

    print("Configuring")
    for channel in range(NUM_CHANNELS):
        rates = sdr.listSampleRates(SoapySDR.SOAPY_SDR_RX, channel)
        sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, channel, rates[1])

        antennas = sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, channel)
        sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, channel, antennas[1])

        sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, channel, 2E6)
        sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, channel, 2402E6)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, channel, "RF", 10)
        sdr.setDCOffsetMode(SoapySDR.SOAPY_SDR_RX, channel, True)

    # allocate buffers and files
    buffs = []
    captures = []
    samples_to_read = CHUNK_SZ * 500
    for i in range(NUM_CHANNELS):
        buffs.append(numpy.zeros(CHUNK_SZ, numpy.complex64))
        captures.append(numpy.zeros(samples_to_read, numpy.complex64))

    print("Setting up stream")
    rxStream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, list(range(NUM_CHANNELS)))
    t_start = time()
    sdr.activateStream(rxStream)

    print("Fetching samples")
    samples_read = 0
    while samples_read < samples_to_read:
        sr = sdr.readStream(rxStream, buffs, CHUNK_SZ)
        for i in range(NUM_CHANNELS):
            captures[i][samples_read:samples_read + sr.ret] = buffs[i][:sr.ret]
            buffs[i][:] = 0
        samples_read += sr.ret
        print("Read %d samples, time %.3f" % (samples_read, sr.timeNs / 1e9), end='\r')
    t_end = time()

    print("Closing stream")
    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)

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
    for b in bursts:
        syms = fsk_decode(b, fs, 1E6, True)
        offset = find_sync32(syms)
        if offset:
            data = unpack_syms(syms, offset)
            data_dw = le_dewhiten(data[4:], 37)
            pkt = le_trim_pkt(data_dw)
            print(hex_str(pkt))
        elif len(b) > 200:
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
