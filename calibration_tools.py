""" Tools for calibration flux density 
using drift scans on Apertif. 

Liam Connor 03/12/2019

Example usage:

If /home/arts/driftscan/3C48/CB00_*_downsamp1000.npy have 
all TABs from CB00 for a 3C48 transit:

python calibration_tools.py ~/driftscan/3C48/CB00_*_downsamp1000.npy --Ndish 8 --src 3C48 --IAB False
"""

import time
import numpy as np

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import glob

import reader

#import psrchive

from APERTIFparams import *
APERTIFparams = APERTIFparams()

class Downsample:
    def __init__(self, chunksize=1e6, ds=10000):
        self.chunksize = int(chunksize)
        self.ds = ds

    def downsample(self, data):
        nt = len(data[0])
        data = data[:, :nt//self.ds*self.ds].reshape(-1, nt//self.ds, self.ds)
        data = np.mean(data, axis=-1)
        
        return data
    
    def downsample_file(self, fn, ds=10000):
        data_ds_full = []
        for ii in xrange(10000000):
            fil_obj = reader.read_fil_data(fn, start=ii*self.chunksize, stop=self.chunksize)[0]
        
            if fil_obj.data.shape[-1] <= ds:
                break
            dt = fil_obj.dt
            data_ds = self.downsample(fil_obj.data)
            data_ds_full.append(data_ds)
            nfreq = data_ds.shape[0]

        print("Downsampled to %f sec" % (ds*dt))
        data_ds_full = np.concatenate(data_ds_full, axis=-1)
        data_ds_full = data_ds_full.reshape(nfreq, -1)
            
        return data_ds_full, ds*dt

class CalibrationTools:

    def __init__(self, t_res=0.8192, freq_up=1550., 
                 freq_low=1250., bw=300.0, 
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
        elif src=='CasA':
            return 5000.0 * (freqMHz / 340.0)**-0.804
        elif src=='TauA':
            return 1000.0 * (freqMHz / 600.0)**-0.2389
        elif src=='3C48':
            # https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance/fdscale
            return 15.4 * (freqMHz / 1500.)**-0.75 
        elif src=='3C196':
            return 13.6 * (freqMHz / 1500.)**-0.8
        else:
            print("Do not recognize source name")
            exit()

    def calculate_snr_rms(self, data_ts, off_samp=(0, 500)):
        """ Calculate S/N at transit of source src
        """

        assert len(data_ts.shape)==1

        off_samp = slice(off_samp[0], off_samp[1])
        noise = data_ts[off_samp][::2] - data_ts[off_samp][1::2]
        sig = np.std(noise)/1.41
        
        # Assume source is not in beam for last 30 samples
        snr = (data_ts.max() - np.median(data_ts[off_samp])) / sig

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
        print "%s is %f Jy at %.1f" % (src, Snu, freq)

        return Tsys


    def calculate_tsys_rms(self, data_ts, freq, off_samp=(0, 500), src='CasA'):
        """ Take ratio of on/off point source data to determine 
        system temperature of instrument. Assume some forward gain 
        and aperture efficiency.
        """

        snr = self.calculate_snr_rms(data_ts, off_samp=off_samp)
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

    def snr_allfreq(self, data, off_samp=(0, 500)):

        snr = []

        for ii, ff in enumerate(self.freq):
            snr.append(self.calculate_snr_rms(data[ii], off_samp=off_samp))

        return np.array(snr)


class Plotter:

    def __init__(self, t_res=0.01, freq_up=1520., freq_low=1220., nfreq=1536):
        self.t_res = t_res
        self.freq_up = freq_up
        self.freq_low = freq_low
        self.freq = np.linspace(freq_up, freq_low, nfreq)
    
    def dyn_spec(self, data):
        tmax = len(data[0])*self.t_res
        data_ = data - np.median(data, axis=-1, keepdims=True)
        data_ /= np.std(data_, axis=-1)[:, None]
        plt.imshow(data_, aspect='auto', 
                   extent=[0, tmax, self.freq_low, self.freq_up],
                   vmax=5, vmin=-1)
        plt.xlabel('Time [s]', fontsize=15)
        plt.ylabel('Freq [MHz]', fontsize=15)

    def plot_ts(self, data, freq_ind):
        ntime = len(data[0])
        times = np.linspace(0, self.t_res*ntime, ntime)
        plt.plot(times, data[freq_ind])
        plt.xlabel('Time [s]', fontsize=15)
        plt.ylabel('Power', fontsize=15)

    def plot_sefd(self, sefd):
        plt.plot(self.freq, sefd, '.')
        plt.xlabel('Freq [MHz]', fontsize=15)
        plt.ylabel('SEFD [Jy]', fontsize=15)
        sefd[sefd!=sefd] = 0.
        plt.ylim(.3*np.median(sefd), 2*np.median(sefd))

    def plot_snr(self, SNR):
        SNR[SNR!=SNR] = 0.
        plt.plot(self.freq, SNR, '.', color='orange')
        plt.xlabel('Freq [MHz]', fontsize=15)
        plt.ylabel('S/N', fontsize=15)
        plt.ylim(.3*np.median(SNR), 3*np.median(SNR))

    def plot_spectrum(self, data):
        plt.plot(self.freq, data[:, -100:].mean(-1))
        plt.xlabel('Freq [MHz]', fontsize=15)
        plt.ylabel('Time-averaged apectrum', fontsize=15)

    def plot_tsys_onoff(self, tsys_onoff):
        plt.plot(self.freq, tsys_onoff)
        plt.xlabel('Freq [MHz]', fontsize=15)
        plt.ylabel('Tsys [K]', fontsize=15)        
        plt.semilogy()

    def plot_all(self, data, sefd, SNR, tsys_onoff, src='CasA', fn='./'):
        fig = plt.figure(figsize=(12,8))

        fig.add_subplot(231)
        self.dyn_spec(data)

        fig.add_subplot(232)
        self.plot_ts(data, len(data)//2)

        fig.add_subplot(233)
        self.plot_spectrum(data)

        fig.add_subplot(234)
        self.plot_sefd(sefd)

        fig.add_subplot(235)
        self.plot_snr(SNR)
        
        fig.add_subplot(236)
        self.plot_tsys_onoff(tsys_onoff)

        plt.suptitle('%s Transit' % src, fontsize=30)

        t0 = time.time()
        fnout = fn.strip('.npy') + '_%s_cal.pdf' % src
        plt.savefig(fnout)
        plt.tight_layout()
        plt.show()
#        plt.show()


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

def run_fluxcal(options, fn):
    data = np.load(fn)
    data[data!=data] = 0.
    nt = data.shape[-1]
    data = data.reshape(-1, 2, nt).mean(1)
    nfreq = data.shape[0]

    try:
        t_res = np.float(fn.split('dt')[-1].split('.npy')[0])
    except:
        t_res = options.t_res

    CalTools = CalibrationTools(t_res=t_res, Ndish=options.Ndish, 
                                IAB=options.IAB, nfreq=nfreq)

    tsys_rms = CalTools.tsys_rms_allfreq(data, off_samp=(0, 200), src=options.src)
    tsys_onoff = CalTools.tsys_onoff_allfreq(data, off_samp=(0, 200), src=options.src)
    sefd_rms = CalTools.tsys_to_sefd(tsys_rms)
    snr = CalTools.snr_allfreq(data, off_samp=(0, 200))

    data_rb = data[:, :data.shape[1]//1*1].reshape(nfreq, -1, 1).mean(-1)

    P = Plotter(t_res=t_res, nfreq=nfreq)
    P.plot_all(data_rb, sefd_rms, snr, tsys_onoff, src=options.src, fn=fn)


if __name__=='__main__':
    import optparse

    parser = optparse.OptionParser(prog="inject_frb.py", \
                        version="", \
                        usage="%prog FN_FILTERBANK FN_FILTERBANK_OUT [OPTIONS]", \
                        description="Create diagnostic plots for individual triggers")

    parser.add_option('--t_res', dest='t_res', type='float', \
                      help="Time resolution in seconds (Default: .8192)", default=0.8192)

    parser.add_option('--IAB', dest='IAB', default=True,\
                      help="Data were taken with incoherent beamforming")

    parser.add_option('--Ndish', dest='Ndish', default=10.0,\
                      help="Number of dishes",
                      type='float')

    parser.add_option('--src', dest='src', default='CasA',\
                      help="Name of source e.g. CasA, TauA, 3C48",
                      type='str')

    options, args = parser.parse_args()

    for fn in args:
        if fn.split('.')[-1]=='fil':
            print("Rebinning and saving to .npy")
            ds = 10000
            D = Downsample()
            data, dt = D.downsample_file(fn, ds=ds)
            fnout = fn.strip('.fil') + '_downsamp%d_dt%.3f' % (ds, dt)
            np.save(fnout, data)
            fn = fnout + '.npy'
            
        run_fluxcal(options, fn)









