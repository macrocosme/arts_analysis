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
    """ Take a path string, combine each file in 
    path along time axis after averaging over 
    pseudo-pulse phase
    """
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
    """ Take ratio of on/off point source data to determine 
    system temperature of instrument. Assume some forward gain 
    and aperture efficiency.
    """
    # Use only Stokes I
    data = data_arr[:, 0]

    # Assume source is not in beam for last 30 samples
    fractional_tsys = data / np.median(data[-30:])
    # Get source flux at this frequency
    Snu = casa_flux(freq)

    G = APERTIFparams.G 
    Tsys = G * Snu / (fractional_tsys - 1)
    return Tsys

def calculate_tsys_allfreq(date, folder, sband=1, eband=16):
    """ Loop over bands from sband to eband, combine 
    in time, then get Tsys as a function of frequency.
    """
    tsys_arr = []
    freq = []
    data_full = []
    nband = int(eband)-int(sband)+1
    fig = plt.figure()

    for band in range(sband, eband+1):
        band = "%02d"%band

        print "subint %s and band %s" % (subints, band)

        fullpath = "/data/%s/Timing/%s/%s" % (band, date, folder)
        filepath = '%s/*%s*.ar' % (fullpath, '_'+subints)
        
        data, cfreq = combine_files_time(filepath)
        data_full.append(data[:, 0]) # Stokes I only
        bw = 18.75
        nsubband = data.shape[-1]
        nfreq = nband * nsubband
        ntimes = data.shape[0]

        # Loop over each sub-band and calculate Tsys
        for nu in xrange(nsubband):
            bfreq = cfreq - bw/2. 
            freqi = bfreq + bw * nu / nsubband
            freq.append(freqi)
            freqind = nsubband * (int(band)-int(sband)) + nu
            tsys = calculate_tsys(data[..., nu], freqi)
            tsys_arr.append(tsys)

    data_full = np.concatenate(data_full)
    tsys_arr = np.concatenate(tsys_arr)

    data_full = data_full.reshape(-1, ntimes, nsubband)
    data_full = data_full.transpose(1, 0, 2).reshape(ntimes, -1)

    tsys_arr.shape = (-1, ntimes)

    plot_tsys_freq(tsys_arr, freq)
#    plotter(tsys_arr, str(cfreq)+'.png')

    return tsys_arr, data_full

def plot_tsys_freq(tsys_arr, freq):
    fig = plt.figure()

    mpix = tsys_arr.shape[-1]//2
    mslice = slice(mpix-5,mpix+5)
    plt.plot(freq, tsys_arr[:, mslice])

    plt.ylim(0, 500)
    plt.show()

def plot_on_off():
    """ Plot Stokes I on source vs. off
    """

    
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

    tsys_arr, data_full = calculate_tsys_allfreq(date, folder, sband=2, eband=16)
    np.save('tsyscasa', tsys_arr)
    np.save('full_data_arr', data_full)


