import numpy as np
import matplotlib.pylab as plt

import argparse
import glob

import psrchive

from APERTIFparams import *
APERTIFparams = APERTIFparams()

def casa_flux(nu_MHz):
    """ Give a frequency in MHz, return 
        a CasA flux in Jy.

        Based on https://arxiv.org/pdf/1609.05940.pdf
    """
    return 5000.0 * (nu_MHz / 340.0)**-0.804


def combine_files_time(fstr):
    flist = glob.glob(fstr)
    flist.sort()

    # This will be the full time-avgd data array
    data_arr = []

    for ff in flist[:]:
        arch = psrchive.Archive_load(ff)
        data = arch.get_data()
        data = data.sum(axis=-1) # Average over pseudo-pulse profile
        data_arr.append(data)

    cfreq = arch.get_centre_frequency()

    data_arr = np.concatenate(data_arr, axis=0)

    return data_arr, cfreq

def calculate_tsys(data_arr, freq):

    # Use only Stokes I
    data = data_arr[:, 0]

    # Assume source is not in beam for last 30 samples
    fractional_tsys = data / np.median(data[-30:])
    # Get source flux at this frequency
    Snu = casa_flux(freq)

    G = APERTIFparams.G 
    Tsys = G * Snu / (fractional_tsys - 1)
    return Tsys

def allfreq(date, folder, sband=1, eband=16):

    tsys_arr = []
    freq = []
    nband = int(eband)-int(sband)+1
    cnames = []
    fig = plt.figure()

    for band in range(sband, eband+1):
        band = "%02d"%band

        print "subint %s and band %s" % (subints, band)

        fullpath = "/data/%s/Timing/%s/%s" % (band, date, folder)
        filepath = '%s/*%s*.ar' % (fullpath, '_'+subints)
        
        data, cfreq = combine_files_time(filepath)

        cnames.append(str(int(cfreq)))
        bw = 18.75
        nsubband = data.shape[-1]
        nfreq = nband * nsubband
        ntimes = data.shape[0]

        plt.plot(data[:, 0].mean(-1)/data[-30:, 0].mean(0).mean(-1))

        for nu in xrange(nsubband):
            bfreq = cfreq - bw/2. 
            freqi = bfreq + bw * nu / nsubband
            freq.append(freqi)
            freqind = nsubband * (int(band)-int(sband)) + nu
            tsys = calculate_tsys(data[..., nu], freqi)
            tsys_arr.append(tsys)

    plt.legend(cnames)
    plt.ylim(0, 10)
    plt.show()

    fig = plt.figure()
    plt.plot(freq, casa_flux(freq))
    plt.show()

    tsys_arr = np.concatenate(tsys_arr)
    np.save('fullarr', tsys_arr)
    print tsys_arr.shape, 'tsys'
    tsys_arr.shape = (-1, ntimes)

    plot_tsys_freq(tsys_arr, freq)
#    plotter(tsys_arr, str(cfreq)+'.png')

    return tsys_arr

def plot_tsys_freq(tsys_arr, freq):
    fig = plt.figure()

    mpix = tsys_arr.shape[-1]//2
    mslice = slice(mpix-5,mpix+5)
    plt.plot(freq, tsys_arr[:, mslice])

    plt.ylim(0, 500)
    plt.show()


# flist = glob.glob('/data/15/Timing/20170302/2017.03.02-10:35:30.B0000+00/2017.03.02-10:35:30.B0000+00.band14_0*.ar') 
# flist.sort()

# data_arr = []

# for ff in flist[:]:
#     arch = psrchive.Archive_load(ff)
#     data = arch.get_data()
#     data = data.sum(axis=-1)
#     data_arr.append(data)

# nfreq = data.shape[-1]
# cfreq = arch.get_centre_frequency()
# bw = 131.25
# freq = np.linspace(cfreq, cfreq + bw, nfreq) - bw/2.
# print freq

# data_arr = np.concatenate(data_arr, axis=0)
# print data_arr.shape

# fig = plt.figure(figsize=(15, 12))

# G = (26 / 64.)**2*0.7

# print G

def plotter(data, outfile):
    fig = plt.figure()

    for i in range(12):
        fig.add_subplot(3,4,i+1)
        data_ = data[:, 12*i:12*i+12].mean(-1)
        plt.plot(data_, '.', lw=3, color='black')
        plt.ylim(0, 5e2)
#        plt.legend([str(np.round(freq[2*i]))+'MHz'])
#        plt.axhline(75.0, linestyle='--', color='red')
#        plt.xlim(40, 140)

        if i % 4 == 0:
            plt.ylabel(r'$T_{sys}$', fontsize=20)

    plt.show()
    plt.savefig(outfile)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="date of observation in yyyymmdd format")
    parser.add_argument("folder", help="subfolder of data/*/Timing/yyyymmdd/\
                                        that contains folded profiles")                                         
    parser.add_argument("-sband", help="start band number", default="1", type=int)  
    parser.add_argument("-eband", help="end band number", default="16", type=int)
    parser.add_argument("-subints", 
           help="only process subints starting with parameter. e.g. 012\
           would analyze only *_012*.ar files", 
           default="")
    parser.add_argument("-manual_dd", help="dedisperse manually", type=int, default=0)
    parser.add_argument("-dm", help="dm for manual dedispersion", type=float, default=0)
    parser.add_argument("-F0", help="pulsar rotation frequency in Hz", type=float, default=1)
    parser.add_argument("-o", help="name of output file name", default="all")

    args = parser.parse_args()

    # Unpack arguments
    date, folder = args.date, args.folder
    sband, eband, outname, subints = args.sband, args.eband, args.o, args.subints

    fstr = '/data/11/Timing/' + date + '/' + folder + '/*.ar'
    print fstr

    tsys_arr = allfreq(date, folder, sband=2, eband=16)
    np.save('tsyscasa', tsys_arr)
    


