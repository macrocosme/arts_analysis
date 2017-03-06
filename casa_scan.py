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

    data_arr = np.concatenate(data_arr, axis=0)

    return data_arr

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

    for band in range(sband, eband+1):
        band = "%02d"%band

        print "subint %s and band %s" % (subints, band)

        fullpath = "/data/%s/Timing/%s/%s" % (band, date, folder)
        filepath = '%s/*%s*.ar' % (fullpath, '_'+subints)
        
        print "Processing total of %d files\n" % len(flist)
        data = combine_files_time(filepath)
        print data.shape

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

# for i in range(12):
#     fig.add_subplot(3,4,i+1)
#     data = data_arr[:, 0, i:i+2].sum(-1)
#     data /= np.median(data[140:])
#     Snu = casa_flux(freq[2*i])

#     plt.plot(G * Snu / (data - 1), '.', lw=3, color='black')
#     plt.ylim(0, 1.2e2)
#     plt.legend([str(np.round(freq[2*i]))+'MHz'])
#     plt.axhline(75.0, linestyle='--', color='red')
#     plt.xlim(40, 140)

#     if i % 4 == 0:
#         plt.ylabel(r'$T_{sys}$', fontsize=20)

# plt.show()
# plt.savefig('here.png')


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

    data = combine_files_time(fstr)   
    allfreq(date, folder, sband=3, eband=13)
    print data.shape
    a = calculate_tsys(data[..., 0], 1500.0)
    


