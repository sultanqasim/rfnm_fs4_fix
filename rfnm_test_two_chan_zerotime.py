#!/usr/bin/env python3

import SoapySDR
import numpy
from time import time

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

        sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, channel, 80E6)
        sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, channel, 2.1E9)
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, channel, 0)

    # allocate buffers and files
    buffs = []
    files = []
    for i in range(NUM_CHANNELS):
        buffs.append(numpy.array([0]*CHUNK_SZ, numpy.complex64))
        files.append(open("test%d.cf32" % i, 'wb'))

    print("Setting up stream")
    rxStream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, list(range(NUM_CHANNELS)))
    t_start = time()
    sdr.activateStream(rxStream)

    print("Fetching samples")
    samples_read = 0
    samples_to_read = CHUNK_SZ * 2000
    while samples_read < samples_to_read:
        sr = sdr.readStream(rxStream, buffs, CHUNK_SZ, timeoutUs=0)
        samples_read += sr.ret
        for i in range(NUM_CHANNELS):
            files[i].write(buffs[i][:sr.ret].tobytes())
            buffs[i][:] = 0
        print("Read %d samples, time %.3f" % (samples_read, sr.timeNs / 1e9), end='\r')
    t_end = time()

    samp_rate = samples_to_read / (t_end - t_start)
    print("\nSample rate: %.3f Msps" % (samp_rate / 1E6))

    print("Closing stream")
    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)

if __name__ == "__main__":
    main()
