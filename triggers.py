import os
import sys

import numpy as np
import scipy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py
import glob
import copy
import optparse

import tools
from pypulsar.formats import filterbank
from pypulsar.formats import spectra

def dm_range(dm_max, dm_min=2., frac=0.2):
    """ Generate list of DM-windows in which 
    to search for single pulse groups. 

    Parameters
    ----------
    dm_max : float 
        max DM 
    dm_min : float  
        min DM 
    frac : float 
        fractional size of each window 

    Returns
    -------
    dm_list : list 
        list of tuples containing (min, max) of each 
        DM window
    """

    dm_list =[]
    prefac = (1-frac)/(1+frac)

    while dm_max>dm_min:
        if dm_max < 20.:
            prefac = 0.0 

        dm_list.append((int(prefac*dm_max), dm_max))
        dm_max = int(prefac*dm_max)

    return dm_list

def read_singlepulse(fn):
    if fn.split('.')[-1] in ('singlepulse', 'txt'):
        A = np.loadtxt(fn)
        dm, sig, tt, downsample = A[:,0], A[:,1], A[:,2], A[:,4]
    elif fn.split('.')[-1]=='trigger':
        A = np.loadtxt(fn)
        dm, sig, tt, downsample = A[:,-2], A[:,-1], A[:, -3], A[:, 3]
    else:
        print("Didn't recognize singlepulse file")
        return 

    if len(A)==0:
        return 0, 0, 0, 0

    return dm, sig, tt, downsample

def get_triggers(fn, sig_thresh=5.0, t_window=0.5, dm_min=0, dm_max=np.inf):
    """ Get brightest trigger in each 10s chunk.

    Parameters
    ----------
    fn : str 
        filename with triggers (.npy, .singlepulse, .trigger)
    sig_thresh : float
        min S/N to include
    t_window : float 
        Size of each time window in seconds

    Returns
    -------
    sig_cut : ndarray
        S/N array of brightest trigger in each DM/T window 
    dm_cut : ndarray
        DMs of brightest trigger in each DM/T window 
    tt_cut : ndarray
        Arrival times of brightest trigger in each DM/T window 
    ds_cut : ndarray 
        downsample factor array of brightest trigger in each DM/T window 
    """

    dm, sig, tt, downsample = read_singlepulse(fn)

    low_sig_ind = np.where(sig < sig_thresh)[0]
    sig = np.delete(sig, low_sig_ind)
    tt = np.delete(tt, low_sig_ind)
    dm = np.delete(dm, low_sig_ind)
    downsample = np.delete(downsample, low_sig_ind)

    sig_cut, dm_cut, tt_cut, ds_cut = [],[],[],[]
    
    tduration = tt.max() - tt.min()
    ntime = int(tduration / t_window)

    # Make dm windows between 90% of the lowest trigger and 
    # 10% of the largest trigger
    dm_list = dm_range(1.1*dm.max(), dm_min=0.9*dm.min())

    print("Grouping in window of %.2f sec" % t_window)
    print("DMs:", dm_list)

    tt_start = tt.min() - .5*t_window

    # might wanna make this a search in (dm,t,width) cubes
    for dms in dm_list:
        for ii in xrange(ntime + 2):
            try:    
                # step through windows of 2 seconds, starting from tt.min()
                t0, tm = t_window*ii + tt_start, t_window*(ii+1) + tt_start
                ind = np.where((dm<dms[1]) & (dm>dms[0]) & (tt<tm) & (tt>t0))[0]
                sig_cut.append(np.amax(sig[ind]))
                dm_cut.append(dm[ind][np.argmax(sig[ind])])
                tt_cut.append(tt[ind][np.argmax(sig[ind])]) 
                ds_cut.append(downsample[ind][np.argmax(sig[ind])])
            except:
                continue
    # now remove the low DM candidates 
    ind = np.where((np.array(dm_cut) >= dm_min) & (np.array(dm_cut) <= dm_max))

    sig_cut = np.array(sig_cut)
    dm_cut = np.array(dm_cut)
    tt_cut = np.array(tt_cut)
    ds_cut = np.array(ds_cut)

    return sig_cut, dm_cut, tt_cut, ds_cut

def get_mask(rfimask, startsamp, N):
    """Return an array of boolean values to act as a mask
        for a Spectra object.

        Inputs:
            rfimask: An rfifind.rfifind object
            startsamp: Starting sample
            N: number of samples to read

        Output:
            mask: 2D numpy array of boolean values. 
                True represents an element that should be masked.
    """
    sampnums = np.arange(startsamp, startsamp+N)
    blocknums = np.floor(sampnums/rfimask.ptsperint).astype('int')
    mask = np.zeros((N, rfimask.nchan), dtype='bool')
    for blocknum in np.unique(blocknums):
        blockmask = np.zeros_like(mask[blocknums==blocknum])
        blockmask[:,rfimask.mask_zap_chans_per_int[blocknum]] = True
        mask[blocknums==blocknum] = blockmask
    return mask.T[::-1]

def plot_three_panel(data_freq_time, data_dm_time, times, dms, 
                     freq_up=1550, freq_low=1250,
                     cmap="RdBu", suptitle="", fnout="out.pdf"):
    figure = plt.figure()
    ax1 = plt.subplot(311)

    plt.imshow(data_freq_time, aspect='auto', vmax=4, vmin=-4, 
               extent=[0, times[-1], freq_low, freq_up], 
               interpolation='nearest', cmap=cmap)
    plt.ylabel('Freq [MHz]')

    plt.subplot(312, sharex=ax1)
    plt.plot(times, data_freq_time.mean(0), color='k')
    plt.ylabel('Flux')

    plt.subplot(313, sharex=ax1)
    plt.imshow(data_dm_time, aspect='auto', 
               extent=[0, times[-1], dms[0], dms[-1]], 
               interpolation='nearest', cmap=cmap)
    plt.xlabel('Time [s]')
    plt.ylabel('DM')

    plt.suptitle(suptitle)

    plt.show()
    plt.savefig(fnout)

def proc_trigger(fn_fil, dm0, t0, sig_cut, 
                 ndm=50, mk_plot=False, downsamp=1, 
                 beamno='', fn_mask=None, nfreq_plot=32,
                 ntime_plot=250,
                 cmap='RdBu'):
    """ Locate data within filterbank file (fn_fi)
    at some time t0, and dedisperse to dm0, generating 
    plots 

    Parameters:
    ----------
    fn_fil     : str 
        name of filterbank file
    dm0        : float 
        trigger dm found by single pulse search software
    t0         : float 
        time in seconds where trigger was found 
    sig_cut    : np.float 
        sigma of detected trigger at (t0, dm0)
    ndm        : int 
        number of DMs to use in DM transform 
    mk_plot    : bool 
        make three-panel plots 
    downsamp   : int 
        factor by which to downsample in time. comes from searchsoft. 
    beamno     : str 
        beam number, for fig names 
    nfreq_plot : int 
        number of frequencies channels to plot 

    Returns:
    -------
    full_dm_arr_downsamp : np.array
        data array with downsampled dm-transformed intensities
    full_freq_arr_downsamp : np.array
        data array with downsampled freq-time intensities 
    """
    rawdatafile = filterbank.filterbank(fn_fil)

    mask = np.array([ 5,   6,   9,  32,  35,  49,  75,  76,  78,  82,  83,  87,  92,
                      93,  97,  98, 108, 110, 111, 112, 114, 118, 122, 123, 124, 157,
                      160, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 660, 
                      661])

    dt = rawdatafile.header['tsamp']
    freq_up = rawdatafile.header['fch1']
    nfreq = rawdatafile.header['nchans']
    freq_low = freq_up + nfreq*rawdatafile.header['foff']
    time_res = dt * downsamp

    dm_min = max(0, dm0-100)
    dm_max = dm0 + 100
    dms = np.linspace(dm_min, dm_max, ndm, endpoint=True)

    # make sure dm0 is in the array
    dm_max_jj = np.argmin(abs(dms-dm0))
    dms += (dm0-dms[dm_max_jj])
    dms[0] = max(0, dms[0])

    # Read in 5 disp delays
    width = 5 * abs(4e3 * dm0 * (freq_up**-2 - freq_low**-2))

    tdisp = width / dt
    tplot = ntime_plot * downsamp 

    if tdisp > tplot:
        # Need to read in more data than you'll plot
        # because of large dispersion time
        chunksize = int(tdisp)
        t_min = chunksize//2 - (ntime_plot*downsamp)//2
        t_max = chunksize//2 + (ntime_plot*downsamp)//2
    else:
        # Only need ot read in enough to plot 
        chunksize = int(tplot)        
        t_min, t_max = 0, chunksize

    start_bin = int(t0/dt - chunksize/2.)

    if start_bin < 0:
        extra = start_bin
        start_bin = 0
        t_min += extra
        t_max += extra

    t_min, t_max = int(t_min), int(t_max)
    ntime = t_max-t_min
    
    full_arr = np.empty([int(ndm), int(ntime)])   

    snr_max = 0

    data = rawdatafile.get_spectra(start_bin, chunksize)
    data.data -= np.median(data.data, axis=-1)[:, None]
    data.data[mask] = 0.

    if not fn_mask is None:
        rfimask = rfifind.rfifind(fn_mask)
        mask = get_mask(rfimask, start_bin, chunksize)
        data = data.masked(mask, maskval='median-mid80')

    for jj, dm_ in enumerate(dms):
        print("Dedispersing to dm=%0.1f at t=%0.1f sec with width=%.2f" % 
                    (dm_, start_bin*dt, downsamp))
        data_copy = copy.deepcopy(data)
        data_copy.dedisperse(dm_)
        dm_arr = data_copy.data[:, t_min:t_max].mean(0)

        # Taken from PRESTO's single_pulse_search:
        # The following gets rid of (hopefully) most of the                                                                                                                
        # outlying values (i.e. power dropouts and single pulses)                                                                                                          
        # If you throw out 5% (2.5% at bottom and 2.5% at top)                                                                                                             
        # of random gaussian deviates, the measured stdev is ~0.871                                                                                                        
        # of the true stdev.  Thus the 1.0/0.871=1.148 correction below.                                                                                                   
        # The following is roughly .std() since we already removed the median 

        # std_chunk = scipy.signal.detrend(dm_arr, type='linear')
        # std_chunk.sort()
        # stds = 1.148*np.sqrt((std_chunk[ntime/40:-ntime/40]**2.0).sum() /
        #                            (0.95*ntime))
        # snr_ = std_chunk[-1]/stds 

        snr_ = tools.calc_snr(dm_arr)

        full_arr[jj] = copy.copy(dm_arr)

        if jj==dm_max_jj:
            data_dm_max = data_copy.data[:, t_min:t_max]

    downsamp = int(downsamp)

    # bin down to nfreq_plot freq channels
    full_freq_arr_downsamp = data_dm_max[:nfreq//nfreq_plot*nfreq_plot, :].reshape(\
                                   nfreq_plot, -1, ntime).mean(1)
    # bin down in time by factor of downsamp
    full_freq_arr_downsamp = full_freq_arr_downsamp[:, :ntime//downsamp*downsamp\
                                   ].reshape(-1, ntime//downsamp, downsamp).mean(-1)
    
    times = np.linspace(0, ntime*dt, len(full_freq_arr_downsamp[0]))

    full_dm_arr_downsamp = full_arr[:, :ntime//downsamp*downsamp]
    full_dm_arr_downsamp = full_dm_arr_downsamp.reshape(-1, 
                                 ntime//downsamp, downsamp).mean(-1)

    full_freq_arr_downsamp /= np.std(full_freq_arr_downsamp)
    full_dm_arr_downsamp /= np.std(full_dm_arr_downsamp)

    suptitle = "beam%s snr%d dm%d t0%d width%d" %\
                 (beamno, sig_cut, dms[dm_max_jj], t0, downsamp)

    fn_fig_out = './plots/train_data_beam%s_snr%d_dm%d_t0%d.pdf' % \
                     (beamno, sig_cut, dms[dm_max_jj], t0)

    if mk_plot is True:
        plot_three_panel(full_freq_arr_downsamp, full_dm_arr_downsamp, 
                         times, dms, freq_low=freq_low, freq_up=freq_up, 
                         suptitle=suptitle, fnout=fn_fig_out)
    
    return full_dm_arr_downsamp, full_freq_arr_downsamp, time_res

def h5_writer(data_freq_time, data_dm_time, 
              dm0, t0, snr, beamno='', basedir='./',
              time_res=''):
    """ Write to an hdf5 file trigger data, 
    pulse parameters
    """
    fnout = '%s/data_trainsnr%d_dm%d_t0%d.hdf5'\
                % (basedir, snr, dm0, t0)

    f = h5py.File(fnout, 'w')
    f.create_dataset('data_freq_time', data=data_freq_time)

    if data_dm_time is not None:    
        f.create_dataset('data_dm_time', data=data_dm_time)
        ndm = data_dm_time.shape[0]
    else:
        ndm = 0

    nfreq, ntime = data_freq_time.shape

    f.attrs.create('snr', data=snr)
    f.attrs.create('dm0', data=dm0)
    f.attrs.create('ndm', data=ndm)
    f.attrs.create('nfreq', data=nfreq)
    f.attrs.create('ntime', data=ntime)
    f.attrs.create('time_res', data=time_res)
    f.attrs.create('t0', data=t0)
    f.attrs.create('beamno', data=beamno)
    f.close()

    print("Wrote to file %s" % fnout)

def file_reader(fn, ftype='hdf5'):
    if ftype is 'hdf5':
        f = h5py.File(fn, 'r')

        data_freq_time = f['data_freq_time'][:]
        data_dm_time = f['data_dm_time'][:]
        attr = f.attrs.items()

        snr, dm0, time_res, t0 = attr[0][1], attr[1][1], attr[5][1], attr[6][1] 

        f.close()

        return data_freq_time, data_dm_time, [snr, dm0, time_res, t0]

    elif ftype is 'npy':
        data = np.load(fn)

        return data

if __name__=='__main__':
# Example usage 
# python triggers.py /data/09/filterbank/20171107/2017.11.07-01:27:36.B0531+21/CB21.fil\
#     CB21_2017.11.07-01:27:36.B0531+21.trigger --sig_thresh 12.0 --mk_plot False

    parser = optparse.OptionParser(prog="triggers.py", \
                        version="", \
                        usage="%prog FN_FILTERBANK FN_TRIGGERS [OPTIONS]", \
                        description="Create diagnostic plots for individual triggers")

    parser.add_option('--sig_thresh', dest='sig_thresh', type='float', \
                        help="Only process events above >sig_thresh S/N" \
                                "(Default: 8.0)", default=8.0)

    parser.add_option('--ndm', dest='ndm', type='int', \
                        help="Number of DMs to use in DM transform (Default: 50).", \
                        default=1)

    parser.add_option('--mask', dest='maskfile', type='string', \
                        help="Mask file produced by rfifind. (Default: No Mask).", \
                        default=None)

    parser.add_option('--save_data', dest='save_data', type='str',
                        help="save each trigger's data. 0=don't save. \
                        hdf5 = save to hdf5. npy=save to npy. concat to \
                        save all triggers into one file",
                        default='hdf5')

    parser.add_option('--ntrig', dest='ntrig', type='int',
                        help="Only process this many triggers",
                        default=None)

    parser.add_option('--mk_plot', dest='mk_plot', action='store_true', \
                        help="make plot if True",
                        default=True)

    parser.add_option('--nfreq_plot', dest='nfreq_plot', type='int',
                        help="make plot with this number of freq channels",
                        default=32)

    parser.add_option('--ntime_plot', dest='ntime_plot', type='int',
                        help="make plot with this number of time samples",
                        default=250)

    parser.add_option('--cmap', dest='cmap', type='str',
                        help="imshow colourmap", 
                        default='RdBu')

    parser.add_option('--dm_min', dest='dm_min', type='float',
                        help="", 
                        default=10.0)

    parser.add_option('--dm_max', dest='dm_max', type='float',
                        help="", 
                        default=np.inf)



    options, args = parser.parse_args()
    fn_fil = args[0]
    fn_sp = args[1]

    if options.save_data == 'concat':
        data_dm_time_full = []
        data_freq_time_full = []
        params_full = []

    sig_cut, dm_cut, tt_cut, ds_cut = tools.get_triggers(fn_sp, 
                                                         sig_thresh=options.sig_thresh,
                                                         dm_min=options.dm_min,
                                                         dm_max=options.dm_max)
    ntrig_grouped = len(sig_cut)
    print("-----------------------------")
    print("Grouped down to %d triggers" % ntrig_grouped)
    print("----------------------------- \n")

    grouped_triggers = np.empty([ntrig_grouped, 4])
    grouped_triggers[:,0] = sig_cut
    grouped_triggers[:,1] = dm_cut
    grouped_triggers[:,2] = tt_cut
    grouped_triggers[:,3] = ds_cut

    np.savetxt('grouped_pulses.singlepulse', grouped_triggers)

    for ii, t0 in enumerate(tt_cut[:options.ntrig]):
        print("\nStarting DM=%0.2f" % dm_cut[ii])
        data_dm_time, data_freq_time, time_res = \
                        proc_trigger(fn_fil, dm_cut[ii], t0, sig_cut[ii],
                        mk_plot=options.mk_plot, ndm=options.ndm, 
                        downsamp=ds_cut[ii], nfreq_plot=options.nfreq_plot,
                        ntime_plot=options.ntime_plot, cmap=options.cmap,
                        fn_mask=options.maskfile)

        basedir = './'

        if options.save_data != '0':
            if options.save_data == 'hdf5':
                h5_writer(data_freq_time, data_dm_time, 
                          dm_cut[ii], t0, sig_cut[ii], 
                          beamno='', basedir=basedir, time_res=time_res)
            elif options.save_data == 'npy':

                fnout_freq_time = '%s/data_trainsnr%d_dm%d_t0%f_freq.npy'\
                         % (basedir, sig_cut[ii], dm_cut[ii], np.round(t0, 2))
                fnout_dm_time = '%s/data_trainsnr%d_dm%d_t0%f_dm.npy'\
                         % (basedir, sig_cut[ii], dm_cut[ii], np.round(t0, 2))

                np.save(fnout_freq_time, data_freqtime)
                np.save(fnout_dm_time, data_dmtime)

            elif options.save_data == 'concat':
                data_dm_time_full.append(data_dm_time)
                data_freq_time_full.append(data_freq_time)
                params = [dm_cut[ii], 0, ds_cut[ii], 0, -2, 0, t0, sig_cut[ii]]
                params_full.append(params)
        else:
            print('Not saving data')

    if options.save_data == 'concat':
        data_dm_time_full = np.concatenate(data_dm_time_full)
        data_freq_time_full = np.concatenate(data_freq_time_full)
        fnout = '%s/data_training_full.hdf5' % basedir

        f = h5py.File(fnout, 'w')
        f.create_dataset('data_freq_time', data=data_freq_time_full)
        f.create_dataset('data_dm_time', data=data_dm_time_full)
        f.create_dataset('params', data=params_full)
        f.close()

        print('Saved all triggers to %s' % fnout)

    exit()








