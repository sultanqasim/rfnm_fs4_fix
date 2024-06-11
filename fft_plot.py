#!/usr/bin/env python3

import numpy
import matplotlib.pyplot as plt
import sys

def main(fname):
    samprate = int(fname[13:-5])
    buf = numpy.fromfile(fname, numpy.complex64)
    plt.psd(buf, 65536, samprate)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1])
