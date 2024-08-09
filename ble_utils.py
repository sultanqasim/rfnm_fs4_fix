import numpy
import scipy
import re
from struct import pack

def burst_detect(capture, thresh=0.002, pad=4):
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

def burst_extract(capture, thresh=0.01, pad=4):
    burst_ranges = burst_detect(capture, thresh, pad)
    ranges = []

    for a, b in burst_ranges:
        ranges.append(capture[a:b])

    return ranges

def squelch(capture, thresh=0.01, pad=4):
    burst_ranges = burst_detect(capture, thresh, pad)
    arr = numpy.zeros(capture.shape, capture.dtype)

    for a, b in burst_ranges:
        arr[a:b] = capture[a:b]

    return arr

def fm_demod(capture):
    phase = numpy.angle(capture)
    d = numpy.diff(phase)
    d += numpy.pi
    d %= 2 * numpy.pi
    d -= numpy.pi
    return d

def fsk_decode(capture, fs, sym_rate, clock_recovery=False, cfo=0):
    demod = fm_demod(capture)

    samps_per_sym = fs / sym_rate
    offset = 0
    if clock_recovery:
        skip = int(samps_per_sym * 3)
        if len(demod) > skip * 2:
            offset = skip + numpy.argmax(demod[skip:skip * 2])

    # convert from Carrier Frequency Offset (CFO) in Hz to radians per sample error
    demod_offset = cfo * 2 * numpy.pi / fs

    indices = numpy.array(numpy.arange(offset, len(capture), samps_per_sym), numpy.int64)
    digital_demod = numpy.array(demod > demod_offset, numpy.uint8)

    """
    fig, axs = plt.subplots(2)
    axs[0].plot(numpy.abs(capture))
    axs[1].plot(demod, ".-", markevery=indices)
    fig.tight_layout()
    """

    return digital_demod[indices]

def find_sync_multi(samples_demod, sync, big_endian=False, corr_thresh=2, samps_per_sym=2):
    if big_endian:
        seq = numpy.unpackbits(numpy.frombuffer(sync, numpy.uint8), bitorder='big')
    else:
        seq = numpy.unpackbits(numpy.frombuffer(sync, numpy.uint8), bitorder='little')

    # make the sequence -1 or +1 so that cross correlation equals number of matching bits
    seq_signed = numpy.zeros(len(seq) * samps_per_sym, dtype=numpy.float32)
    seq_signed[0::samps_per_sym] = ((2 * seq) - 1).view(numpy.int8)
    syms_signed = (2 * samples_demod.view(numpy.int8)) - 1

    corr = numpy.correlate(syms_signed, seq_signed)
    peaks, _ = scipy.signal.find_peaks(corr, len(seq) - corr_thresh)

    return peaks

def find_sync_multi2(samples_demod, sync, samps_per_sym=2):
    indices = []

    sync_len = len(sync) * 8
    sync_bits = numpy.unpackbits(numpy.frombuffer(sync, numpy.uint8), bitorder='little')
    sync_seqs = [numpy.packbits(sync_bits[8-i:sync_len-i], bitorder='little').tobytes() for i in range(8)]

    for i in range(samps_per_sym):
        syms = numpy.packbits(samples_demod[i::samps_per_sym], bitorder='little').tobytes()
        for j, seq in enumerate(sync_seqs):
            indices.extend([((m.start() - 1)*8 + j)*samps_per_sym + i for m in re.finditer(seq, syms)])

    # deduplicate
    indices.sort()
    last_index = -samps_per_sym
    indices2 = []
    for i in indices:
        if i - last_index < samps_per_sym: continue
        indices2.append(i)
        last_index = i

    return indices2

def ble_pkt_extract(samples_demod, peaks, chan, samps_per_sym=2):
    pkts = []
    MAX_PKT = 264 # 4 byte AA, 2 byte header, 255 byte body, 3 byte CRC
    for p in peaks:
        syms = samples_demod[p:p+(8*MAX_PKT*samps_per_sym):samps_per_sym]
        raw = unpack_syms(syms, 32)
        if len(raw) > 2:
            hdr = le_dewhiten(raw[:2], chan)
            pkt_len = 5 + hdr[1]
            pkts.append(le_dewhiten(raw[:pkt_len], chan))
    return pkts

def find_sync(syms, sync: bytes, big_endian=False, corr_thresh=2):
    if big_endian:
        seq = numpy.unpackbits(numpy.frombuffer(sync, numpy.uint8), bitorder='big')
    else:
        seq = numpy.unpackbits(numpy.frombuffer(sync, numpy.uint8), bitorder='little')

    # make the sequences -1 or +1 so that cross correlation equals number of matching bits
    seq_signed = ((2 * seq) - 1).view(numpy.int8)
    syms_signed = ((2 * syms) - 1).view(numpy.int8)
    corr = numpy.correlate(syms_signed, seq_signed)
    pos = numpy.argmax(corr)
    if corr[pos] >= len(seq) - corr_thresh:
        return pos
    else:
        return None

def find_sync32(syms, sync_word=0x8E89BED6):
    sync = pack('<I', sync_word)
    return find_sync(syms, sync)

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
