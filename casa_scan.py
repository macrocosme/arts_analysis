import numpy as np
import matplotlib.pylab as plt

import argparse
import glob

import psrchive

from APERTIFparams import *
APERTIFparams = APERTIFparams()

class CalibrationTools:

    def __init__(self, t_res=0.01, freq_up=1520., 
                 freq_low=1220., bw=300.0, 
                 nfreq=1536, Ndish=10, IAB=True):

        self.chan_width = bw / nfreq * 1e6
        self.t_res = t_res
        self.freq_up = freq_up
        self.freq_low = freq_low
        self.freq = np.linspace(freq_up, freq_low, nfreq)
        self.Ndish = Ndish
        self.IAB = IAB

    def calc_gain(self):
        gain_parkes = 0.7 # gain of Parkes K Jy**-1
        gain = (25./64.)**2*self.Ndish*gain_parkes

        if self.IAB is True:
            gain /= np.sqrt(self.Ndish)

        return gain

    def source_flux(self, freqMHz, src='CasA'):
        """ Give a frequency in MHz, return 
            a CasA flux in Jy.

            Based on https://arxiv.org/pdf/1609.05940.pdf
        """
        if src is None:
            print("No source name given")
            return 1.0
        elif src is 'CasA':
            return 5000.0 * (freqMHz / 340.0)**-0.804
        elif src is 'TauA':
            return 1000.0 * (freqMHz / 600.0)**-0.2389


    def calculate_snr_rms(self, data_ts, off_samp=(0, 500), src='CasA'):
        """ Calculate S/N at transit of source src
        """

        assert len(data_ts.shape)==1

        off_samp = slice(off_samp[0], off_samp[1])

        # Assume source is not in beam for last 30 samples
        snr = (data_ts.max() - np.median(data_ts[off_samp])) / np.std(data_ts[off_samp])

        return snr

    def calculate_tsys_onoff(self, data_ts, freq, off_samp=(0, 500), src='CasA'):
        """ Take ratio of on/off point source data to determine 
        system temperature of instrument. Assume some forward gain 
        and aperture efficiency.
        """

        assert len(data_ts.shape)==1

        off_samp = slice(off_samp[0], off_samp[1])

        # Assume source is not in beam for last 30 samples
        fractional_tsys = data_ts.max() / np.median(data_ts[off_samp])
        # Get source flux at this frequency
        Snu = self.source_flux(freq, src=src)

        G = self.calc_gain()
        Tsys = G * Snu / (fractional_tsys - 1)
        print "%s is %f Jy" % (src, Snu)

        return Tsys


    def calculate_tsys_rms(self, data_ts, freq, off_samp=(0, 500), src='CasA'):
        """ Take ratio of on/off point source data to determine 
        system temperature of instrument. Assume some forward gain 
        and aperture efficiency.
        """

        snr = self.calculate_snr_rms(data_ts, off_samp=off_samp, src=src)

        # Get source flux at this frequency
        Snu = self.source_flux(freq, src=src)

        G = self.calc_gain()
        Tsys = G * Snu * np.sqrt(self.chan_width*self.t_res) / snr

        print "%s is %f Jy" % (src, Snu)

        return Tsys

    def tsys_to_sefd(self, Tsys):
        G = self.calc_gain()

        return Tsys / G

    def tsys_rms_allfreq(self, data, off_samp=(0, 500), src='CasA'):

        tsys_rms = []

        for ii, ff in enumerate(self.freq):
            tsys_rms.append(self.calculate_tsys_rms(data[ii], ff, off_samp=off_samp, src=src))

        return np.array(tsys_rms)

    def tsys_onoff_allfreq(self, data, off_samp=(0, 500), src='CasA'):

        tsys_onoff = []

        for ii, ff in enumerate(self.freq):
            tsys_onoff.append(self.calculate_tsys_onoff(data[ii], ff, off_samp=off_samp, src=src))

        return np.array(tsys_onoff)


class Plotter:

    def __init__(self, t_res=0.01, freq_up=1520., freq_low=1220., nfreq=1536):
        self.t_res = t_res
        self.freq_up = freq_up
        self.freq_low = freq_low
        self.freq = np.linspace(freq_up, freq_low, nfreq)
    
    def dyn_spec(self, data):
        tmax = len(data[0])*self.t_res
        plt.imshow(data, aspect='auto', extent=[0, tmax, self.freq_up, self.freq_low])
        plt.xlabel('Time [s]')
        plt.ylabel('Freq [MHz]')

    def plot_ts(self, data, freq_ind):
        ntime = len(data[0])
        times = np.linspace(0, self.t_res*ntime, ntime)
        plt.plot(times, data[freq_ind])
        plt.xlabel('Time [s]')

    def plot_sefd(self, sefd):
        plt.plot(self.freq, sefd)
        plt.xlabel('Freq [MHz]', fontsize=15)
        plt.ylabel('SEFD [Jy]', fontsize=15)

    def plot_snr(self, SNR):
        plt.plot(self.freq, SNR, color='orange')
        plt.xlabel('Freq [MHz]', fontsize=15)
        plt.ylabel('S/N', fontsize=15)

    def plot_all(self, data, sefd, SNR):
        fig = plt.figure(figsize=(12,12))

        fig.add_subplot(221)
        self.dyn_spec(data)

        fig.add_subplot(222)
        self.plot_ts(data, 300)

        fig.add_subplot(223)
        self.plot_sefd(sefd)

        fig.add_subplot(224)
        self.plot_snr(SNR)

        plt.show()


def source_flux(nu_MHz, src=None):
    """ Give a frequency in MHz, return 
        a CasA flux in Jy.

        Based on https://arxiv.org/pdf/1609.05940.pdf
    """
    if src is None:
        return 1.0
    elif src is 'CasA':
        return 5000.0 * (nu_MHz / 340.0)**-0.804
    elif src is 'TauA':
        return 1000.0 * (nu_MHz / 600.0)**-0.2389


def combine_files_time(fstr):
    """ Take a path string, combine each file in 
    path along time axis after averaging over 
    pseudo-pulse phase
    """
    flist = glob.glob(fstr)
    flist.sort()

    # This will be the full time-avgd data array
    data_arr = []

    assert len(flist) > 0

    for ff in flist[:]:
        arch = psrchive.Archive_load(ff)
        data = arch.get_data()
        data = data.sum(axis=-1) # Average over pseudo-pulse profile
        data_arr.append(data)

    cfreq = arch.get_centre_frequency()

    data_arr = np.concatenate(data_arr, axis=0)

    return data_arr, cfreq

def calculate_tsys(data_arr, freq, src='CasA'):
    """ Take ratio of on/off point source data to determine 
    system temperature of instrument. Assume some forward gain 
    and aperture efficiency.
    """
    # Use only Stokes I
    data = data_arr[:, 0]

    # Assume source is not in beam for last 30 samples
    fractional_tsys = data / np.median(data[-30:])
    # Get source flux at this frequency
    Snu = source_flux(freq, src=src)

    G = APERTIFparams.G * np.sqrt(2)
    Tsys = G * Snu / (fractional_tsys - 1)
    print "%s is %f Jy" % (src, Snu)
    return Tsys



def calculate_tsys_allfreq(date, folder, sband=1, eband=16, src='CasA'):
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

        print "Reading band %s" % band

        fullpath = "/data/%s/Timing/%s/%s" % (band, date, folder)
        filepath = '%s/*.ar' % fullpath
        
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
            tsys = calculate_tsys(data[..., nu], freqi, src=src)
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
    return


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

    parser = optparse.OptionParser(prog="inject_frb.py", \
                        version="", \
                        usage="%prog FN_FILTERBANK FN_FILTERBANK_OUT [OPTIONS]", \
                        description="Create diagnostic plots for individual triggers")

    parser.add_option('--t_res', dest='t_res', type='float', \
                      help="Time resolution in seconds", \
                                "(Default: .01)", default=0.01)

    parser.add_option('--IAB', dest='IAB', default=True,\
                      help="Data were taken with incoherent beamforming")

    parser.add_option('--Ndish', dest='Ndish', default=10.0,\
                      help="Number of dishes",
                      type='float')

    
    parser.add_option('--dm_list', type='string', action='callback', callback=foo_callback)

    options, args = parser.parse_args()
    fn = args[0]

    data = np.load(fn)
    nfreq = data.shape[0]

    CalTools = CalibrationTools(t_res=options.t_res, Ndish=options.Ndish, 
                                IAB=options.IAB)

    tsys_rms = CalibrationTools.tsys_onoff_allfreq(data, off_samp=(0, 5000), src='CasA')
    tsys_onoff = CalibrationTools.tsys_onoff_allfreq(data, off_samp=(0, 5000), src='CasA')
    sefd_rms = CalibrationTools.tsys_to_sefd(tsys_rms)

    # Rebin in time by x100 before plotting
    data_rb = data[:, :data.shape[1]//100*100].reshape(nfreq, -1, 100).mean(-1)

    Plotter = Plotter(t_res=options.t_res*100)
    Plotter.plot_all(data_rb, snr, C.tsys_to_sefd(tsys))


# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("date", help="date of observation in yyyymmdd format")
#     parser.add_argument("folder", help="subfolder of data/*/Timing/yyyymmdd/\
#                                         that contains folded profiles")                                         
#     parser.add_argument("-sband", help="start band number", default="1", type=int)  
#     parser.add_argument("-eband", help="end band number", default="16", type=int)
#     parser.add_argument("-subints", 
#            help="only process subints starting with parameter. e.g. 012\
#            would analyze only *_012*.ar files", 
#            default="")
#     parser.add_argument("-manual_dd", help="dedisperse manually", type=int, default=0)
#     parser.add_argument("-dm", help="dm for manual dedispersion", type=float, default=0)
#     parser.add_argument("-F0", help="pulsar rotation frequency in Hz", type=float, default=1)
#     parser.add_argument("-o", help="name of output file name", default="all")
#     parser.add_argument("src", help="source name", default="CasA")

#     args = parser.parse_args()

#     # Unpack arguments
#     date, folder = args.date, args.folder
#     sband, eband, outname, subints, src = args.sband, args.eband, args.o, args.subints, args.src

#     tsys_arr, data_full = calculate_tsys_allfreq(date, folder, sband=2, eband=16, src=src)
#     np.save('tsyscasa', tsys_arr)
#     np.save('full_data_arr', data_full)


