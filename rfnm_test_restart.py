#!/usr/bin/env python3

import SoapySDR
import numpy

def main():
    print("Opening RFNM")
    args = dict(driver="RFNM")
    sdr = SoapySDR.Device(args)

    print("Configuring")
    rates = sdr.listSampleRates(SoapySDR.SOAPY_SDR_RX, 0)
    print("Setting sample rate", rates[0])
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, rates[0])

    antennas = sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)
    print("Setting antenna", antennas[0])
    sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, antennas[0])

    sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, 80E6)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, 2.1E9)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 0)

    for i in range(3):
        print("Setting up stream")
        rxStream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])
        sdr.activateStream(rxStream)

        print("Fetching samples")
        CHUNK_SZ = 1 << 18
        buff = numpy.array([0]*CHUNK_SZ, numpy.complex64)
        for i in range(200):
            sr = sdr.readStream(rxStream, [buff], CHUNK_SZ)
            if sr.ret < CHUNK_SZ:
                print("ERROR: Read timeout!")
                break
            else:
                print("Got chunk %d" % i, end='\r')

        print("Closing stream")
        sdr.deactivateStream(rxStream)
        sdr.closeStream(rxStream)

if __name__ == "__main__":
    main()
