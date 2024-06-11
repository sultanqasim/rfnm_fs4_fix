#!/usr/bin/env python3

import numpy
import matplotlib.pyplot as plt
import sys

def main(fname):
    samprate = int(fname[13:-5])
    buf = numpy.fromfile(fname, numpy.complex64)

    PSD_SIZE = 8192

    plt.figure(figsize=[8, 12], dpi=240)
    fig, axs = plt.subplots(3)
    fig.set_dpi(240)
    fig.set_size_inches(6, 8)

    axs[0].psd(buf, PSD_SIZE, samprate)
    axs[0].set_title("Original")

    buf_dc = buf - numpy.mean(buf)
    axs[1].psd(buf_dc, PSD_SIZE, samprate)
    axs[1].set_title("DC Offset Corrected")

    buf_dc4 = numpy.copy(buf)
    for i in range(4):
        m = numpy.mean(buf_dc4[i::4])
        buf_dc4[i::4] -= m
    axs[2].psd(buf_dc4, PSD_SIZE, samprate)
    axs[2].set_title("Quad DC Offset Corrected")

    fig.tight_layout()
    fig.savefig("rfnm_dc_correction_%d.png" % samprate, bbox_inches="tight")

if __name__ == "__main__":
    main(sys.argv[1])
