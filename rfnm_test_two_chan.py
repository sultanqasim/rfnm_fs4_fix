#!/usr/bin/env python3

import SoapySDR
import numpy

NUM_CHANNELS = 2
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
    for i in range(5000):
        sr = sdr.readStream(rxStream, buffs, CHUNK_SZ, timeoutUs=20000)
        if sr.ret < CHUNK_SZ:
            print("\nERROR: Read timeout!")
            break
        else:
            print("Got chunk %d, time %.3f" % (i, sr.timeNs / 1e9), end='\r')

    print("Closing stream")
    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)

if __name__ == "__main__":
    main()
