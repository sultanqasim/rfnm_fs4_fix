#!/usr/bin/env python3

import SoapySDR
import numpy
from time import time
import matplotlib.pyplot as plt

NUM_CHANNELS = 2
CHUNK_SZ = 1 << 18

def main():
    print("Opening RFNM")
    args = dict(driver="RFNM")
    sdr = SoapySDR.Device(args)

    print("Configuring")
    for channel in range(NUM_CHANNELS):
        rates = sdr.listSampleRates(SoapySDR.SOAPY_SDR_RX, channel)
        sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, channel, rates[1])

        antennas = sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, channel)
        sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, channel, antennas[0])

        sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, channel, 2E6)
        sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, channel, 2402E6)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, channel, "RF", 15)
        sdr.setDCOffsetMode(SoapySDR.SOAPY_SDR_RX, channel, True)

    # allocate buffers and files
    buffs = []
    captures = []
    samples_to_read = CHUNK_SZ * 50
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
        sr = sdr.readStream(rxStream, buffs, CHUNK_SZ, timeoutUs=0)
        for i in range(NUM_CHANNELS):
            captures[i][samples_read:samples_read + sr.ret] = buffs[i][:sr.ret]
            buffs[i][:] = 0
        samples_read += sr.ret
        print("Read %d samples, time %.3f" % (samples_read, sr.timeNs / 1e9), end='\r')
    t_end = time()

    samp_rate = samples_to_read / (t_end - t_start)
    print("\nSample rate: %.3f Msps" % (samp_rate / 1E6))

    print("Closing stream")
    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)

    print("Plotting")
    fig, axs = plt.subplots(2)
    fig.set_dpi(240)
    fig.set_size_inches(6, 8)
    axs[0].plot(numpy.abs(captures[0]))
    axs[1].plot(numpy.abs(captures[1]))
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
