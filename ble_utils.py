import numpy
import scipy
from struct import pack

def burst_detect(capture, thresh=0.002, pad=10):
    mag_low = numpy.abs(capture) > thresh * 0.8
    mag_high = numpy.abs(capture) > thresh * 1.2

    ranges = []
    x = 0
    while x < len(capture):
        start = x + numpy.argmax(mag_high[x:])
        if start == x and not mag_high[x]:
            break
        stop = start + numpy.argmin(mag_low[start:])
        if stop == start and mag_low[-1]:
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

def burst_extract(capture, thresh=0.01, pad=10):
    burst_ranges = burst_detect(capture, thresh, pad)
    ranges = []

    for a, b in burst_ranges:
        ranges.append(capture[a:b])

    return ranges

def squelch(capture, thresh=0.01, pad=10):
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
    print("mean", numpy.mean(demod))
    demod -= numpy.mean(demod) # CFO correction

    offset = 0
    if clock_recovery:
        skip = int(samps_per_sym * 3)
        offset = skip + numpy.argmax(demod[skip:skip * 2])

    indices = numpy.array(numpy.arange(offset, len(capture), samps_per_sym), numpy.int64)
    digital_demod = numpy.array(demod > 0, numpy.uint8)

    """
    fig, axs = plt.subplots(2)
    axs[0].plot(numpy.abs(capture))
    axs[1].plot(demod, ".-", markevery=indices)
    fig.tight_layout()
    """

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

whitening = [
    1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, # 0
	1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, # 16
	1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, # 32
	1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, # 48
	0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, # 64
	0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, # 80
	0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, # 96
	1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1     # 112
]

whitening_index = [
    70, 62, 120, 111, 77, 46, 15, 101, 66, 39, 31, 26, 80,
    83, 125, 89, 10, 35, 8, 54, 122, 17, 33, 0, 58, 115, 6,
    94, 86, 49, 52, 20, 40, 27, 84, 90, 63, 112, 47, 102
]

# code based on ubertooth
def le_dewhiten(data, chan):
	dw = []
	idx = whitening_index[chan]

	for b in data:
		o = 0
		for i in range(8):
			bit = (b >> i) & 1
			bit ^= whitening[idx]
			idx = (idx + 1) % len(whitening)
			o |= bit << i
		dw.append(o)

	return bytes(dw)

def le_trim_pkt(data):
    # 2 bytes header, n byte body, 3 byte CRC
    l = 2 + data[1] + 3
    return data[:l]
