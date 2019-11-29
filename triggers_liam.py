import os
import sys
import time

import numpy as np
import scipy
import matplotlib as mpl
mpl.use('Agg', warn=False)
import matplotlib.pyplot as plt
import h5py
import glob
import copy
import optparse
import logging
from threading import Thread

from pypulsar.formats import filterbank, spectra

try:
    from darc.sb_generator import SBGenerator
    HAVE_SB = True
except ImportError:
    HAVE_SB = False

import tools
import plotter
import realtime_tools

RTproc = realtime_tools.RealtimeProc()

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


def multiproc_dedisp(dm):
    datacopy.dedisperse(dm)
    data_freq_time = datacopy[:, t_min:t_max]

    return (datacopy.data.mean(0), data_freq_time)


def sys_temperature_bandpass(data):
    """Bandpass calibrate based on system temperature.
    The lowest noise way to flatten the bandpass. Very good if T_sys is
    relatively constant accross the band.
    """

    T_sys = np.median(data, 1)
    bad_chans = T_sys < 0.001 * np.median(T_sys)
    T_sys[bad_chans] = 1
    data /= T_sys[:,None]
    data[bad_chans,:] = 0


def remove_noisy_freq(data, sigma_threshold):
    """Flag frequency channels with high variance.
    To be effective, data should be bandpass calibrated in some way.
    """

    nfreq = data.shape[0]
    ntime = data.shape[1]

    # Calculate variances without making full data copy (as numpy does).
    var = np.empty(nfreq, dtype=np.float64)
    skew = np.empty(nfreq, dtype=np.float64)
    kurt = np.empty(nfreq, dtype=np.float64)
    for ii in range(nfreq):
        var[ii] = np.var(data[ii,:])
        skew[ii] = np.mean((data[ii,:] - np.mean(data[ii,:])**3))
        kurt[ii] = np.mean((data[ii,:] - np.mean(data[ii,:])**4))

    # Find the bad channels.
    bad_chans = False
    for ii in range(3):
        bad_chans_var = abs(var - np.mean(var)) > sigma_threshold * np.std(var)
        bad_chans_skew = abs(skew - np.mean(skew)) > sigma_threshold * np.std(skew)
        bad_chans_kurt = abs(kurt - np.mean(kurt)) > sigma_threshold * np.std(kurt)
        bad_chans = np.logical_or(bad_chans, bad_chans_var)
        bad_chans = np.logical_or(bad_chans, bad_chans_skew)
        bad_chans = np.logical_or(bad_chans, bad_chans_kurt)
        var[bad_chans] = np.mean(var)
        skew[bad_chans] = np.mean(skew)
        kurt[bad_chans] = np.mean(kurt)
    data[bad_chans,:] = 0


def remove_noisy_channels(data, sigma_threshold=2, iters=10):
    """Flag frequency channels with high variance.
    To be effective, data should be bandpass calibrated in some way.
    """
    var = np.var(data, axis=1)

    for ii in range(10):
        var[np.abs(var-np.median(var))>sigma_threshold*np.std(var)] = 0

    bad_chans = np.where(var==0)[0]
    data[bad_chans] = 0.

    return data


def cleandata(data, threshold_time=3.25, threshold_frequency=2.75, bin_size=32,
              n_iter_time=3, n_iter_frequency=3, clean_type='time'):
    """ Take filterbank object and mask
    RFI time samples with average spectrum.

    Parameters:
    ----------
    data :
        data array (nfreq, ntime)
    threshold_time : float
        units of sigma
    threshold_frequency : float
        units of sigma
    bin_size : int
        quantization bin size
    n_iter_time : int
        Number of iteration for time cleaning
    n_iter_frequency : int
        Number of iteration for frequency cleaning
    clean_type : str
        type of cleaning to be done.
        Accepted values: 'time', 'frequency', 'both', 'perchannel'

    Returns:
    -------
    cleaned filterbank object
    """
    if clean_type not in ['time', 'both', 'frequency', 'perchannel']:
        logging.info("Not RFI cleaning. Expected time, both, perchannel or frequency as clean_type")
        return data
        
    logging.info("Cleaning RFI")

    dtmean = np.mean(data, axis=-1)
    # Clean in time
    #sys_temperature_bandpass(data.data)
    #remove_noisy_freq(data.data, 3)
    #remove_noisy_channels(data.data, sigma_threshold=2, iters=5)
    if clean_type in ['time', 'both']:
        for i in range(n_iter_time):
            dfmean = np.mean(data, axis=0)
            stdevf = np.std(dfmean)
            medf = np.median(dfmean)
            maskf = np.where(np.abs(dfmean - medf) > threshold_time*stdevf)[0]
            # replace with mean spectrum
            data[:, maskf] = dtmean[:, None]*np.ones(len(maskf))[None]

    if clean_type=='perchannel':
        for ii in range(n_iter_time):
            dtmean = np.mean(data, axis=1, keepdims=True)
            dtsig = np.std(data, axis=1)
            for nu in range(data.shape[0]):
                d = dtmean[nu]
                sig = dtsig[nu]
                maskpc = np.where(np.abs(data[nu]-d)>threshold_time*sig)[0]
                data[nu][maskpc] = d

    # Clean in frequency
    # remove bandpass by averaging over bin_size ajdacent channels
    if clean_type in ['frequency', 'both']:
        for ii in range(n_iter_frequency):
            dtmean_nobandpass = data.mean(1) - dtmean.reshape(-1, bin_size).mean(-1).repeat(bin_size)
            stdevt = np.std(dtmean_nobandpass)
            medt = np.median(dtmean_nobandpass)
            maskt = np.abs(dtmean_nobandpass - medt) > threshold_frequency*stdevt
            data[maskt] = np.median(dtmean)#dtmean.reshape(-1, bin_size).mean(-1).repeat(bin_size)[maskt]

    return data

def load_tab_data(fname, start_bin, chunksize, out=None, tab=None):
    f = filterbank.filterbank(fname)
    if out is not None:
        out[tab] = f.get_spectra(start_bin, chunksize).data
        return
    else:
        data = f.get_spectra(start_bin, chunksize).data
    f.close()
    return data

def proc_trigger(fn_fil, dm0, t0, sig_cut,
                 ndm=50, mk_plot=False, downsamp=1,
                 beamno='', fn_mask=None, nfreq_plot=32,
                 ntime_plot=250,
                 cmap='RdBu', cand_no=1, multiproc=False,
                 rficlean=False, snr_comparison=-1,
                 outdir='./', sig_thresh_local=5.0,
                 subtract_zerodm=False,
                 threshold_time=3.25, threshold_frequency=2.75, bin_size=32,
                 n_iter_time=3, n_iter_frequency=3, clean_type='time', freq=1370,
                 sb_generator=None, sb=None):
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
    freq       : int
        central frequency used to find zapped channels file
    sb_generator: SBGenerator object
        synthesized beam mapper from DARC (None for TAB/IAB)
    sb         : int
        synthesized beam to generate (None for TAB/IAB)

    Returns:
    -------
    full_dm_arr_downsamp : np.array
        data array with downsampled dm-transformed intensities
    full_freq_arr_downsamp : np.array
        data array with downsampled freq-time intensities
    """

    try:
        rfimask = np.loadtxt('/home/arts/.controller/amber_conf/zapped_channels_{:.0f}.conf'.format(freq))
        rfimask = rfimask.astype(int)
    except:
        rfimask = np.array([])
        logging.warning("Could not load dumb RFIMask")

    SNRtools = tools.SNR_Tools()
    downsamp = min(4096, downsamp)

    # store path to filterbanks
    if sb is not None:
        prefix_fil = fn_fil
        # get first file
        fn_fil = prefix_fil + '_00.fil'
    rawdatafile = filterbank.filterbank(fn_fil)
    dfreq_MHz = rawdatafile.header['foff']

    dt = rawdatafile.header['tsamp']
    freq_up = rawdatafile.header['fch1']
    nfreq = rawdatafile.header['nchans']
    # fix RFI mask order
    rfimask = nfreq - rfimask
    freq_low = freq_up + nfreq*rawdatafile.header['foff']
    ntime_fil = (os.path.getsize(fn_fil) - rawdatafile.header_size)/nfreq
    tdm = np.abs(8.3*1e-6*dm0*dfreq_MHz*(freq_low/1000.)**-3)

    dm_min = max(0, dm0-40)
    dm_max = dm0 + 40
    dms = np.linspace(dm_min, dm_max, ndm, endpoint=True)

    # make sure dm0 is in the array
    dm_max_jj = np.argmin(abs(dms-dm0))
    dms += (dm0-dms[dm_max_jj])
    dms[0] = max(0, dms[0])

    global t_min, t_max
    # if smearing timescale is < 4*pulse width,
    # downsample before dedispersion for speed
    downsamp_smear = max(1, int(downsamp*dt/tdm/2.))
    # ensure that it's not larger than pulse width
    downsamp_smear = int(min(downsamp, downsamp_smear))
    downsamp_res = int(downsamp//downsamp_smear)
    downsamp = int(downsamp_res*downsamp_smear)
    time_res = dt * downsamp
    logging.info("Width_full:%d  Width_smear:%d  Width_res: %d" %
                 (downsamp, downsamp_smear, downsamp_res))

    start_bin = int(t0/dt - ntime_plot*downsamp//2)
    width = abs(4.148e3 * dm0 * (freq_up**-2 - freq_low**-2))
    chunksize = int(width/dt + ntime_plot*downsamp)

    t_min, t_max = 0, ntime_plot*downsamp

    if start_bin < 0:
        extra = start_bin
        start_bin = 0
        t_min += extra
        t_max += extra

    t_min, t_max = int(t_min), int(t_max)

    snr_max = 0

    # Account for the pre-downsampling to speed up dedispersion
    t_min /= downsamp_smear
    t_max /= downsamp_smear
    ntime = t_max-t_min

    if ntime_fil < (start_bin+chunksize):
        logging.info("Trigger at end of file, skipping")
#        print("Trigger at end of file, skipping")
        return [], [], [], []

    # get data of all files (SB) or one file (TAB/IAB)
    if sb is not None:
        ntab = 12
        data = np.zeros((ntab, nfreq, chunksize))
        # get list of unique TABs in required SB
        sb_map = list(set(sb_generator.get_map(sb)))
        logging.info("SB {} consists of TABs {}".format(sb, sb_map))
        threads = []
        for tab in range(ntab):
            # skip if we do not need this TAB
            if not tab in sb_map:
                continue
            fname = prefix_fil + '_{:02d}.fil'.format(tab)
            load_tab_data(fname, start_bin, chunksize, out=data, tab=tab)
            #thread = Thread(target=load_tab_data, args=[fname, start_bin, chunksize], kwargs={'out': data, 'tab': tab}, name='TAB{}'.format(tab))
            #thread.daemon = True
            #thread.start()
            #threads.append(thread)
        for thread in threads:
            logging.info("Waiting for loading of {}".format(thread.name))
            thread.join()
        # generate sb
        logging.info("Synthesizing beam {}".format(sb))

        data = sb_generator.synthesize_beam(data, sb=sb)
        # convert to a spectra object, mimicking filterbank.get_spectra
        data = spectra.Spectra(rawdatafile.frequencies, rawdatafile.tsamp, data,
                               starttime=start_bin*rawdatafile.tsamp, dm=0)
    else:
        data = rawdatafile.get_spectra(start_bin, chunksize)

    rawdatafile.close()
    # apply dumb mask
    data.data[rfimask] = 0.

    if rficlean is True:
        data.data = cleandata(data.data, threshold_time, threshold_frequency, bin_size, \
                         n_iter_time, n_iter_frequency, clean_type)

    if subtract_zerodm:
        data.data -= np.mean(data.data, axis=0, keepdims=True)

    freq_ref = 0.5*(freq_up+freq_low)
    # Downsample before dedispersion up to 1/4th
    # DM smearing limit
    data.downsample(downsamp_smear)
    data.data -= np.median(data.data, axis=-1)[:, None]
#    full_arr = np.empty([int(ndm), int(ntime)])
    if not fn_mask is None:
        pass
        # rfimask = rfifind.rfifind(fn_mask)
        # mask = get_mask(rfimask, start_bin, chunksize)
        # data = data.masked(mask, maskval='median-mid80')

    if multiproc is True:
        pass
    else:
        logging.info("\nDedispersing Serially\n")
        data_copy = copy.deepcopy(data)
        data_copy.dedisperse(dm0)

        data_dm_max = data_copy.data[:, max(0, t_min):t_max]
        snr_max = SNRtools.calc_snr_matchedfilter(data_dm_max.mean(0), widths=[downsamp_res])[0]

        if t_min<0:
            Z = np.zeros([nfreq, np.abs(t_min)])
            data_dm_max = np.concatenate([Z, data_dm_max], axis=1)

        # scale max DM by pulse width, 5 units for each ms 
        dm_max_trans = 10. + 5*time_res/0.001 + 10*dm0/1000.
        dm_min_trans = -10. - 5*time_res/0.001 - 10*dm0/1000.

        if dm0+dm_min_trans<=0:
            dm_min_trans = 0.
            dm_max_trans = 2*dm0
            dm_center = dm0
        else:
            dm_center = 0.

        full_arr, dms, times = RTproc.dm_transform(data_dm_max, 
                                                    (freq_up, freq_low), dt=dt*downsamp_smear, 
                                                       dm_max=dm_max_trans, dm_min=dm_min_trans, freq_ref=freq_ref,ndm=ndm, dm0=dm_center)
        dms += dm0

        for jj, dm_ in enumerate(dms):
            continue
            data_copy = copy.deepcopy(data)

            t0_dm = time.time()
            data_copy.dedisperse(dm_)
            dm_arr = data_copy.data[:, max(0, t_min):t_max].mean(0)

            full_arr[jj, np.abs(min(0, t_min)):] = copy.copy(dm_arr)

            logging.info("Dedispersing to dm=%0.1f at t=%0.1fsec with width=%.1f S/N=%.1f" %
                         (dm_, t0, downsamp, sig_cut))

            if jj==dm_max_jj:
                data_dm_max = data_copy.data[:, max(0, t_min):t_max]
                snr_max = SNRtools.calc_snr_matchedfilter(data_dm_max.mean(0), widths=[downsamp_res])[0]
                if t_min<0:
                    Z = np.zeros([nfreq, np.abs(t_min)])
                    data_dm_max = np.concatenate([Z, data_dm_max], axis=1)

    # bin down to nfreq_plot freq channels
    full_freq_arr_downsamp = data_dm_max[:nfreq//nfreq_plot*nfreq_plot, :].reshape(\
                                   nfreq_plot, -1, ntime).mean(1)

    # bin down in time by factor of downsamp
    full_freq_arr_downsamp = full_freq_arr_downsamp[:, :ntime//downsamp_res*downsamp_res\
                                   ].reshape(-1, ntime//downsamp_res, downsamp_res).mean(-1)

    if snr_max < sig_thresh_local:
        logging.info("\nSkipping trigger below local threshold %.2f:" % sig_thresh_local)
        logging.info("snr_local=%.2f  snr_trigger=%.2f\n" % (snr_max, sig_cut))
        return [], [], [], []

    times = np.linspace(0, ntime_plot*downsamp*dt, len(full_freq_arr_downsamp[0]))

    full_dm_arr_downsamp = full_arr[:, :ntime//downsamp_res*downsamp_res]
    full_dm_arr_downsamp = full_dm_arr_downsamp.reshape(-1,
                             ntime//downsamp_res, downsamp_res).mean(-1)

    full_freq_arr_downsamp /= np.std(full_freq_arr_downsamp)
    full_dm_arr_downsamp /= np.std(full_dm_arr_downsamp)

    suptitle = " CB:%s  S/N$_{pipe}$:%.1f  S/N$_{presto}$:%.1f\
                 S/N$_{compare}$:%.1f \nDM:%d  t:%.1fs  width:%d" %\
                 (beamno, sig_cut, snr_max, snr_comparison,
                    dm0, t0, downsamp)

    if not os.path.isdir('%s/plots' % outdir):
        os.system('mkdir -p %s/plots' % outdir)

    if sb is None:
        sbname = -1
    else:
        sbname = sb

    fn_fig_out = '%s/plots/CB%s_snr%d_dm%d_t0%d_sb%d.pdf' % \
                     (outdir, beamno, sig_cut, dm0, t0, sbname)

    params = sig_cut, dm0, downsamp, t0, dt
    tmed = np.median(full_freq_arr_downsamp, axis=-1, keepdims=True)
    full_freq_arr_downsamp -= tmed

    if mk_plot is True:
        logging.info(fn_fig_out)

        if ndm == 1:
            plotter.plot_two_panel(full_freq_arr_downsamp, params, prob=None,
                                   freq_low=freq_low, freq_up=freq_up,
                                   cand_no=cand_no, times=times, suptitle=suptitle,
                                   fnout=fn_fig_out)
        else:
            plotter.plot_three_panel(full_freq_arr_downsamp,
                                     full_dm_arr_downsamp, params, dms,
                                     times=times, freq_low=freq_low,
                                     freq_up=freq_up,
                                     suptitle=suptitle, fnout=fn_fig_out,
                                     cand_no=cand_no)

    return full_dm_arr_downsamp, full_freq_arr_downsamp, time_res, params


def h5_writer(data_freq_time, data_dm_time,
              dm0, t0, snr, beamno='', basedir='./',
              time_res=''):
    """ Write to an hdf5 file trigger data,
    pulse parameters
    """
    fnout = '%s/CB%s_snr%d_dm%d_t0%d.hdf5'\
                % (basedir, beamno, snr, dm0, t0)

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

    logging.info("Wrote to file %s" % fnout)


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
    # The following would generate synthesized beams, rfi clean the data, and plot each trigger
    # python /home/arts/ARTS-obs/external/arts-analysis/triggers_liam.py --rficlean --sig_thresh_local 3 --time_limit 18000 --descending_snr --beamno 00 --dm_min 20 --dm_max 5000 --sig_thresh 30.0 --ndm 64 --save_data concat --nfreq_plot 32 --ntime_plot 64 --cmap viridis --outdir=/data2/output/20190711/2019-07-11-04:03:00.FRB190709/triggers --clean_type time --synthesized_beams --sbmin 0 --sbmax 35 --central_freq 1370 /data2/output/20190711/2019-07-11-04:03:00.FRB190709/filterbank/CB00 /data2/output/20190711/2019-07-11-04:03:00.FRB190709/amber/CB00.trigger

    parser = optparse.OptionParser(prog="triggers.py",
                                   version="",
                                   usage="%prog FN_FILTERBANK_PREFIX FN_TRIGGERS [OPTIONS]",
                                   description="Create diagnostic plots for individual triggers")

    parser.add_option('--sig_thresh', dest='sig_thresh', type='float',
                      help="Only process events above >sig_thresh S/N" \
                           "(Default: 8.0)", default=8.0)

    parser.add_option('--sig_max', dest='sig_max', type='float',
                      help="Only process events above <sig_max S/N" \
                           "(Default: 8.0)", default=np.inf)

    parser.add_option('--ndm', dest='ndm', type='int',
                      help="Number of DMs to use in DM transform (Default: 50).",
                      default=1)

    parser.add_option('--mask', dest='maskfile', type='string',
                      help="Mask file produced by rfifind. (Default: No Mask).",
                      default=None)

    parser.add_option('--save_data', dest='save_data', type='str',
                      help="save each trigger's data. 0=don't save. \
                      hdf5 = save to hdf5. npy=save to npy. concat to \
                      save all triggers into one file",
                      default='hdf5')

    parser.add_option('--ntrig', dest='ntrig', type='int',
                      help="Only process this many triggers",
                      default=None)

    parser.add_option('--mk_plot', dest='mk_plot', action='store_true',
                      help="make plot if True (default False)", default=False)

    parser.add_option('--multiproc', dest='multiproc', action='store_true',
                      help="use multicores if True (default False)", default=False)

    parser.add_option('--rficlean', dest='rficlean', action='store_true',
                      help="use rficlean if True (default False)", default=False)

    parser.add_option('--threshold_time', dest='threshold_time', type=float,
                      help="If rficlean is True, defines threshold for time-domain clean (default 3.25)",
                      default=3.25)

    parser.add_option('--threshold_frequency', dest='threshold_frequency', type=float,
                      help="If rficlean is True, defines threshold for freqency-domain clean (default 2.5)",
                      default=2.75)

    parser.add_option('--bin_size', dest='bin_size', action='store_true',
                      help="If rficlean is True, defines bin size for bandpass removal (default 32)",
                      default=32)

    parser.add_option('--n_iter_time', dest='n_iter_time', type=int,
                      help="If rficlean is True, defines number of iteration for time-domain clean (default 3)",
                      default=3)

    parser.add_option('--n_iter_frequency', dest='n_iter_frequency', type=int,
                      help="If rficlean is True, defines number of iteration for frequency-domain clean (default 3)",
                      default=3)

    parser.add_option('--clean_type', dest='clean_type',
                      help="If rficlean is True, defines type of clean (default 'time')",
                      choices=['time', 'freqency', 'both', 'perchannel'], default='time')

    parser.add_option('--subtract_zerodm', dest='subtract_zerodm', action='store_true',
                      help="use DM=0 timestream subtraction if True (default False)", default=False)

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

    parser.add_option('--time_limit', dest='time_limit', type='float',
                      help="Total time to spend processing in seconds",
                      default=np.inf)

    parser.add_option('--dm_max', dest='dm_max', type='float',
                      help="",
                      default=np.inf)

    parser.add_option('--sig_thresh_local', dest='sig_thresh_local', type='float',
                      help="",
                      default=0.0)

    parser.add_option('--outdir', dest='outdir', type='str',
                      help="directory to write data to",
                      default='./data/')

    parser.add_option('--compare_trig', dest='compare_trig', type='str',
                      help="Compare input triggers with another trigger file",
                      default=None)

    parser.add_option('--beamno', dest='beamno', type='str',
                      help="Beam number of input data",
                      default='')

    parser.add_option('--descending_snr', dest='descending_snr', action='store_true',
                      help="Process from highest to lowest S/N if True (default False)", default=False)

    parser.add_option('--tab', dest='tab', type=int,
                      help="TAB to process (0 for IAB) (default: 0)", default=0)

    parser.add_option('--synthesized_beams', dest='sb', action='store_true',
                      help="Process synthesized beams")

    parser.add_option('--sbmin', type=int, default=0, help="First SB to process data for (Default: 0)")

    parser.add_option('--sbmax', type=int, default=70, help="Last SB to process data for (Default: 70)")

    parser.add_option('--central_freq', dest='freq', type=int, default=1370, 
                      help="Central frequency in zapped channels filename (Default: 1370)")

    logfn = time.strftime("%Y%m%d-%H%M") + '.log'
    logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO, filename=logfn)

    start_time = time.time()

    options, args = parser.parse_args()

    # Explicitly disable TAB if SB is enabled
    if options.sb:
        options.tab = None
    else:
        options.tab_str = "_{:02d}".format(options.tab)

    # TAB: filterbank filename is specified
    # SB: filterbank prefix is specified (i.e. without _<TABno>.fil)
    fn_fil = args[0]
    fn_sp = args[1]

    if options.save_data == 'concat':
        data_dm_time_full = []
        data_freq_time_full = []
        params_full = []
        if options.sb:
            data_sb_full = []

    if options.multiproc is True:
        import multiprocessing

    SNRTools = tools.SNR_Tools()

    if options.compare_trig is not None:
        par_1, par_2, par_match_arr, ind_missed, ind_matched = SNRTools.compare_snr(
                                        fn_sp, options.compare_trig,
                                        dm_min=options.dm_min,
                                        dm_max=options.dm_max,
                                        save_data=False,
                                        sig_thresh=options.sig_thresh,
                                        max_rows=None,
                                        t_window=0.25)

        snr_1, snr_2 = par_1[0], par_2[0]
        snr_comparison_arr = np.zeros_like(snr_1)
        ind_missed = np.array(ind_missed)
        snr_comparison_arr[ind_matched] = par_match_arr[0, :, 1]
        sig_cut, dm_cut, tt_cut, ds_cut, ind_full = par_1[0], par_1[1], \
                                par_1[2], par_1[3], par_1[4]
    else:
        # in SB mode, do grouping over all SBs, then process only the given ones
        if options.sb:
            sig_cut, dm_cut, tt_cut, ds_cut, sb_cut, ind_full = tools.get_triggers(fn_sp, sig_thresh=options.sig_thresh,
                                                                                   dm_min=options.dm_min,
                                                                                   dm_max=options.dm_max,
                                                                                   sig_max=options.sig_max,
                                                                                   t_window=0.5, read_beam=True)
        else:
            sig_cut, dm_cut, tt_cut, ds_cut, ind_full = tools.get_triggers(fn_sp, sig_thresh=options.sig_thresh,
                                                                           dm_min=options.dm_min,
                                                                           dm_max=options.dm_max,
                                                                           sig_max=options.sig_max,
                                                                           t_window=0.5, tab=options.tab)

    if options.descending_snr:
        sig_index = np.argsort(sig_cut)[::-1]
        sig_cut = sig_cut[sig_index]
        dm_cut = dm_cut[sig_index]
        tt_cut = tt_cut[sig_index]
        ds_cut = ds_cut[sig_index]
        ind_full = ind_full[sig_index]
        if options.sb:
            sb_cut = sb_cut[sig_index]

    ntrig_grouped = len(sig_cut)
    logging.info("-----------------------------\nGrouped down to %d triggers" % ntrig_grouped)

    logging.info("DMs: %s" % dm_cut)
    logging.info("S/N: %s" % sig_cut)

    if options.sb:
        grouped_triggers = np.empty([ntrig_grouped, 5])
    else:
        grouped_triggers = np.empty([ntrig_grouped, 4])
    grouped_triggers[:, 0] = sig_cut
    grouped_triggers[:, 1] = dm_cut
    grouped_triggers[:, 2] = tt_cut
    grouped_triggers[:, 3] = ds_cut
    if options.sb:
        grouped_triggers[:, 4] = sb_cut

    ndm = options.ndm
    nfreq_plot = options.nfreq_plot
    ntime_plot = options.ntime_plot
    basedir = options.outdir + '/data/'

    if not os.path.isdir(basedir):
        os.system('mkdir -p %s' % basedir)

    # only if SB 0 is included, write the grouped pulses file to file to avoid writing it several times
    # beam number is only included as last column in SB case
    if options.sb and (options.sbmin == 0):
        np.savetxt(options.outdir + '/grouped_pulses_synthesized_beams.singlepulse',
                   grouped_triggers, fmt='%0.2f %0.1f %0.3f %0.1f %0.0f')
    elif not options.sb:
        np.savetxt(options.outdir+'/grouped_pulses{}.singlepulse'.format(options.tab_str),
                    grouped_triggers, fmt='%0.2f %0.1f %0.3f %0.1f')

    # if SB, only select triggers in given sb range
    if options.sb:
        mask = np.where((sb_cut >= options.sbmin) & (sb_cut <= options.sbmax))
        grouped_triggers = grouped_triggers[mask]
        sig_cut, dm_cut, tt_cut, ds_cut, sb_cut = grouped_triggers.T
        ntrig_grouped = len(sig_cut)
        logging.info("-----------------------------\nGrouped down to %d triggers after SB selection" % ntrig_grouped)
        print("-----------------------------\nGrouped down to %d triggers after SB selection" % ntrig_grouped)

        logging.info("DMs: %s" % dm_cut)
        logging.info("S/N: %s" % sig_cut)

    skipped_counter = 0
    ii = None

    # Initalize SB generator
    if options.sb:
        sb_generator = SBGenerator.from_science_case(science_case=4)
        sb_generator.reversed = True
    else:
        sb_generator = None

    for ii, t0 in enumerate(tt_cut[:options.ntrig]):
        try:
            snr_comparison = snr_comparison_arr[ii]
        except:
            snr_comparison = -1

        if options.sb:
            sb = int(sb_cut[ii])
            logging.info("\n%d/%d\nStarting DM=%0.2f S/N=%0.2f width=%d time=%f sb=%d" % (ii+1, len(tt_cut[:options.ntrig]), dm_cut[ii], sig_cut[ii], ds_cut[ii],
                                                                                   t0, sb))
        else:
            sb = None
            logging.info("\n%d/%d\nStarting DM=%0.2f S/N=%0.2f width=%d time=%f" % (ii+1, len(tt_cut[:options.ntrig]), dm_cut[ii], sig_cut[ii], ds_cut[ii], t0))

        data_dm_time, data_freq_time, time_res, params = proc_trigger(
                                        fn_fil, dm_cut[ii], t0, sig_cut[ii],
                                        mk_plot=options.mk_plot, ndm=options.ndm,
                                        downsamp=ds_cut[ii], nfreq_plot=options.nfreq_plot,
                                        ntime_plot=options.ntime_plot, cmap=options.cmap,
                                        fn_mask=options.maskfile, cand_no=ii,
                                        multiproc=options.multiproc,
                                        rficlean=options.rficlean,
                                        snr_comparison=snr_comparison,
                                        outdir=options.outdir,
                                        beamno=options.beamno,
                                        sig_thresh_local=options.sig_thresh_local,
                                        subtract_zerodm=options.subtract_zerodm,
                                        threshold_time=options.threshold_time,
                                        threshold_frequency=options.threshold_frequency,
                                        bin_size=options.bin_size,
                                        n_iter_time=options.n_iter_time,
                                        n_iter_frequency=options.n_iter_frequency,
                                        clean_type=options.clean_type,
                                        sb_generator=sb_generator,
                                        sb=sb, freq=options.freq)

        if len(data_dm_time)==0:
            skipped_counter += 1
            continue

        if options.save_data != '0':
            if options.save_data == 'hdf5':
                if options.sb:
                    fnout = '{}/data_sb{:02d}_{:02d}_full.hdf5'.format(basedir, options.sbmin, options.sbmax)
                else:
                    fnout = '%s/data%s_full.hdf5' % (basedir, options.tab_str)
                
                f = h5py.File(fnout, 'w')
                f.create_dataset('data_freq_time', data=data_freq_time_full)
                f.create_dataset('data_dm_time', data=data_dm_time_full)
                f.create_dataset('params', data=params_full)
                f.create_dataset('ntriggers_skipped', data=[skipped_counter])
                if options.tab is not None:
                    f.create_dataset('tab', data=np.int(options.tab)*np.ones([len(data_freq_time_full)]))
                elif options.sb:
                    f.create_dataset('sb', data=data_sb_full)
                f.close()
#                h5_writer(data_freq_time, data_dm_time,
#                          dm_cut[ii], t0, sig_cut[ii],
#                          beamno=options.tab_str, basedir=basedir, time_res=time_res)
            elif options.save_data == 'npy':
                fnout_freq_time = '%s/data%s_snr%d_dm%d_t0%f_freq.npy'\
                         % ( basedir, options.tab_str, sig_cut[ii], dm_cut[ii], np.round(t0, 2))
                fnout_dm_time = '%s/data%s_snr%d_dm%d_t0%f_dm.npy'\
                         % (basedir, options.tab_str, sig_cut[ii], dm_cut[ii], np.round(t0, 2))

                np.save(fnout_freq_time, data_freq_time)
                np.save(fnout_dm_time, data_dm_time)

            elif options.save_data == 'concat':
                data_dm_time_full.append(data_dm_time)
                data_freq_time_full.append(data_freq_time)
                params_full.append(params)
                if options.sb:
                    data_sb_full.append(sb)
        else:
            logging.info('Not saving data')

        time_elapsed = time.time() - start_time

        if time_elapsed > options.time_limit:
            logging.info("Exceeded time limit. Breaking loop.")
            break

    if options.save_data == 'concat':
        if len(data_freq_time_full)==0:
            print("\nFound no triggers to concat\n")
            data_dm_time_full = []
            data_freq_time_full = []
            params_full = []
            if options.sb:
                data_sb_full = []
        else:
            if len(data_dm_time_full)==1:
                data_dm_time_full = np.array(data_dm_time_full)
                data_freq_time_full = np.array(data_freq_time_full)
            else:
                data_dm_time_full = np.concatenate(data_dm_time_full, axis=0)
                data_freq_time_full = np.concatenate(data_freq_time_full, axis=0)

            data_dm_time_full = data_dm_time_full.reshape(-1,ndm,ntime_plot)
            data_freq_time_full = data_freq_time_full.reshape(-1,nfreq_plot,ntime_plot)

        if options.sb:
            data_sb_full = np.array(data_sb_full)
            fnout = '{}/data_sb{:02d}_{:02d}_full.hdf5'.format(basedir, options.sbmin, options.sbmax)
        else:
            fnout = '%s/data%s_full.hdf5' % (basedir, options.tab_str)

        f = h5py.File(fnout, 'w')
        f.create_dataset('data_freq_time', data=data_freq_time_full)
        f.create_dataset('data_dm_time', data=data_dm_time_full)
        f.create_dataset('params', data=params_full)
        f.create_dataset('ntriggers_skipped', data=[skipped_counter])
        if options.tab is not None:
            f.create_dataset('tab', data=np.int(options.tab)*np.ones([len(data_freq_time_full)]))
        elif options.sb:
            f.create_dataset('sb', data=data_sb_full)
        f.close()

        logging.info('Saved all triggers to %s' % fnout)
    if ii is not None:
        logging.warning("Skipped %d out of %d triggers" % (skipped_counter, ii+1))
    else:
        logging.warning("There were no triggers")

    exit()
