#!/usr/bin/env python3

import SoapySDR
import numpy
import scipy.signal
from time import time
import matplotlib.pyplot as plt
from struct import pack, unpack

NUM_CHANNELS = 1
CHUNK_SZ = 1 << 18

def burst_detect(capture, thresh=0.02, pad=10):
    mag_high = numpy.abs(capture) > thresh

    ranges = []
    x = 0
    while x < len(capture):
        start = x + numpy.argmax(mag_high[x:])
        stop = start + numpy.argmin(mag_high[start:])
        if start == x:
            break
        if stop == start:
            stop = len(capture)
        start -= pad
        stop += pad
        if start < 0:
            start = 0
        if stop > len(capture):
            stop = len(capture)
        ranges.append((start, stop))
        x = stop

    return ranges

def burst_extract(capture, thresh=0.02, pad=10):
    burst_ranges = burst_detect(capture, thresh, pad)
    ranges = []

    for a, b in burst_ranges:
        ranges.append(capture[a:b])

    return ranges

def squelch(capture, thresh=0.02, pad=10):
    burst_ranges = burst_detect(capture, thresh, pad)
    arr = numpy.zeros(capture.shape, capture.dtype)

    for a, b in burst_ranges:
        arr[a:b] = capture[a:b]

    return arr

def fm_demod(capture):
    phase = numpy.angle(capture)
    return numpy.gradient(numpy.unwrap(phase))

def fsk_decode(capture, samps_per_sym, clock_recovery=False):
    demod = fm_demod(capture)

    offset = 0
    if clock_recovery:
        skip = int(samps_per_sym * 2)
        offset = skip + numpy.argmax(demod[skip:skip * 2])

    indices = numpy.array(numpy.arange(offset, len(capture), samps_per_sym), numpy.int64)
    digital_demod = numpy.array(demod > 0, numpy.uint8)

    return digital_demod[indices]

def find_sync32(syms, sync_word=0x8E89BED6):
    seq = numpy.unpackbits(numpy.frombuffer(pack('<I', sync_word), numpy.uint8), bitorder='little')
    found = False
    i = 0
    while i < len(syms) - 32:
        if numpy.array_equal(syms[i:i+32], seq):
            found = True
            break
        i += 1

    if found:
        return i
    else:
        return None

def unpack_syms(syms, start_offset):
    return numpy.packbits(syms[start_offset:], bitorder='little')

def hex_str(b):
    chars = ["%02x" % c for c in b]
    return " ".join(chars)

def main():
    print("Opening RFNM")
    args = dict(driver="RFNM")
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
    samples_to_read = CHUNK_SZ * 100
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
    fs = sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, channel)
    lpf = scipy.signal.butter(3, 1E6, fs=fs)
    capf = scipy.signal.lfilter(*lpf, captures[0])
    ds = capf[::8]
    fs //= 8

    print("Bursts")
    bursts = burst_extract(ds)

    print("Demod")
    samps_per_sym = fs / 1E6
    for b in bursts:
        syms = fsk_decode(b, samps_per_sym, True)
        offset = find_sync32(syms)
        if offset:
            data = unpack_syms(syms, offset)
            print(hex_str(data))
        else:
            print("sync not found")

    """
    print("Plotting")
    fig, axs = plt.subplots(1)
    axs.plot(fm_demod(squelch(ds)))
    fig.tight_layout()
    plt.show()
    """

if __name__ == "__main__":
    main()
