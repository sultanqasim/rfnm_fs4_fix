#!/usr/bin/env python3

import SoapySDR
import numpy
import sys

def main():
    print("Opening RFNM")
    args = dict(driver="rfnm")
    sdr = SoapySDR.Device(args)

    print("Configuring")
    rates = sdr.listSampleRates(SoapySDR.SOAPY_SDR_RX, 0)
    print("Setting sample rate", rates[0])
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, rates[0])

    antennas = sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)
    print("Setting antenna", antennas[1])
    sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, antennas[1])

    sdr.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, 90E6)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, 2440E6)
    sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, "RF", 10)
    sdr.setDCOffsetMode(SoapySDR.SOAPY_SDR_RX, 0, True)

    print("Setting up stream")
    rxStream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [0])
    sdr.activateStream(rxStream)

    print("Fetching samples")
    CHUNK_SZ = 1 << 18
    buff = numpy.array([0]*CHUNK_SZ, numpy.complex64)
    f = open("ble_capture_f_2440_sr_%d.cf32" % rates[0], 'wb')
    for i in range(2000):
        sr = sdr.readStream(rxStream, [buff], CHUNK_SZ)
        if sr.ret < CHUNK_SZ:
            print("ERROR: Read timeout!")
            break
        else:
            print("Got chunk %d" % i, end='\r')
            f.write(buff[:sr.ret].tobytes())

    print("\nClosing")
    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)

if __name__ == "__main__":
    main()
