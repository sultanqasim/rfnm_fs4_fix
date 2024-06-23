#!/usr/bin/env python3

import SoapySDR
import numpy

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

    print("Setting up stream")
    rxStream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, list(range(NUM_CHANNELS)))
    sdr.activateStream(rxStream)

    # allocate buffers
    buffs = []
    for i in range(NUM_CHANNELS):
        buffs.append(numpy.array([0]*CHUNK_SZ, numpy.complex64))

    print("Fetching samples")
    samples_read = 0
    while samples_read < CHUNK_SZ * 2000:
        sr = sdr.readStream(rxStream, buffs, CHUNK_SZ, timeoutUs=0)
        samples_read += sr.ret
        print("Read %d samples" % samples_read, end='\r')

    print("Closing stream")
    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)

if __name__ == "__main__":
    main()
