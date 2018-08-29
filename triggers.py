import os
import sys
import time

import numpy as np
import scipy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py
import glob
import copy
import optparse

from pypulsar.formats import filterbank

import tools
import plotter 


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

def proc_trigger(fn_fil, dm0, t0, sig_cut, 
                 ndm=50, mk_plot=False, downsamp=1, 
                 beamno='', fn_mask=None, nfreq_plot=32,
                 ntime_plot=250,
                 cmap='RdBu', cand_no=1, multiproc=False,
                 rficlean=False, snr_comparison=-1,
                 outdir='./'):
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
    SNRtools = tools.SNR_Tools()
    downsamp = min(512, downsamp)
    rawdatafile = filterbank.filterbank(fn_fil)
    dfreq_MHz = rawdatafile.header['foff']    
    mask = []

    dt = rawdatafile.header['tsamp']
    freq_up = rawdatafile.header['fch1']
    nfreq = rawdatafile.header['nchans']
    freq_low = freq_up + nfreq*rawdatafile.header['foff']
    ntime_fil = (os.path.getsize(fn_fil) - 467.)/nfreq
    tdm = np.abs(8.3*1e-6*dm0*dfreq_MHz*(freq_low/1000.)**-3)

    dm_min = max(0, dm0-40)
    dm_max = dm0 + 40
    dms = np.linspace(dm_min, dm_max, ndm, endpoint=True)

    # make sure dm0 is in the array
    dm_max_jj = np.argmin(abs(dms-dm0))
    dms += (dm0-dms[dm_max_jj])
    dms[0] = max(0, dms[0])

    # Read in 2 disp delays
#    width = 2.5*abs(4.14e3 * dm0 * (freq_up**-2 - freq_low**-2))

#    tdisp = width / dt

    global t_min, t_max
    # if smearing timescale is < 4*pulse width, 
    # downsample before dedispersion for speed 
    downsamp_smear = max(1, int(downsamp*dt/tdm/4.))
    # ensure that it's not larger than pulse width
    downsamp_smear = int(min(downsamp, downsamp_smear))
    downsamp_res = int(downsamp//downsamp_smear)
    downsamp = int(downsamp_res*downsamp_smear)
    time_res = dt * downsamp
    tplot = ntime_plot * downsamp 
    print("Width_full:%d  Width_smear:%d  Width_res: %d" % 
        (downsamp, downsamp_smear, downsamp_res))

    # if tdisp > tplot:
    #     # Need to read in more data than you'll plot
    #     # because of large dispersion time
    #     chunksize = int(tdisp)
    #     t_min = chunksize//2 - (ntime_plot*downsamp)//2
    #     t_max = chunksize//2 + (ntime_plot*downsamp)//2
    # else:
    #     # Only need to read in enough to plot 
    #     chunksize = int(tplot)        
    #     t_min, t_max = 0, chunksize

#    start_bin = int(t0/dt - chunksize/2.)

    start_bin = int(t0/dt - ntime_plot*downsamp//2)
    width = abs(4.14e3 * dm0 * (freq_up**-2 - freq_low**-2))
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
        print("Trigger at end of file, skipping")
        return [],[],[],[]

    print("Reading in chunk: %d" % chunksize)
    data = rawdatafile.get_spectra(start_bin, chunksize)
    
    if rficlean is True:
        print("Cleaning RFI")
        data_tmean = np.mean(data.data, axis=-1)
        data_nobandpass = data_tmean - data_tmean.reshape(-1, 16).mean(-1).repeat(16)
        stdev, med = SNRtools.sigma_from_mad(data_nobandpass)
        mask = np.where(np.abs(data_nobandpass - med) > 5.0*stdev)[0]
        data.data[mask] = 0.0

    # Downsample before dedispersion up to 1/4th 
    # DM smearing limit 
    data.downsample(downsamp_smear)
    data.data -= np.median(data.data, axis=-1)[:, None]
    full_arr = np.empty([int(ndm), int(ntime)])   

    if not fn_mask is None:
        pass
        # rfimask = rfifind.rfifind(fn_mask)
        # mask = get_mask(rfimask, start_bin, chunksize)
        # data = data.masked(mask, maskval='median-mid80')

    if multiproc is True:
        tbeg=time.time()
        global datacopy 

        size_arr = sys.getsizeof(data.data)
        nproc = int(32.0e9/size_arr)

        ndm_ = min(min(nproc, ndm), 10)

        print("\nDedispersing in Parallel with %d proc\nAssuming 50GB of memory" % ndm_)
        for kk in range(ndm//ndm_):
            dms_ = dms[ndm_*kk:ndm_*(kk+1)]
            datacopy = copy.deepcopy(data)
            pool = multiprocessing.Pool(processes=ndm_)        
            data_tuple = pool.map(multiproc_dedisp, [i for i in dms_])
            pool.close()

            data_tuple = np.concatenate(data_tuple)
            ddm = np.concatenate(data_tuple[0::2]).reshape(ndm_, -1)
            df = np.concatenate(data_tuple[1::2]).reshape(ndm_, nfreq, -1)

            print(time.time()-tbeg)
            full_arr[ndm_*kk:ndm_*(kk+1)] = ddm[:, t_min:t_max]

            ind_kk = range(ndm_*kk, ndm_*(kk+1))

            if dm_max_jj in ind_kk:
                print(dms_[ind_kk.index(dm_max_jj)])
                data_dm_max = df[ind_kk.index(dm_max_jj)]#dm_max_jj]hack

            del ddm, df
    else:
        print("\nDedispersing Serially\n")
        tbeg = time.time()
        for jj, dm_ in enumerate(dms):
            data_copy = copy.deepcopy(data)

            data_copy.dedisperse(dm_)
            dm_arr = data_copy.data[:, max(0, t_min):t_max].mean(0)
            snr_ = SNRtools.calc_snr(dm_arr)
            full_arr[jj, np.abs(min(0, t_min)):] = copy.copy(dm_arr)

            print("Dedispersing to dm=%0.1f at t=%0.1fsec with width=%.1f S/N=%.1f" % 
                        (dm_, t0, downsamp, snr_))

            if jj==dm_max_jj:
                snr_max = snr_
                data_dm_max = data_copy.data[:, max(0, t_min):t_max]

                if t_min<0:
                    Z = np.zeros([nfreq, np.abs(t_min)])
                    data_dm_max = np.concatenate([Z, data_dm_max], axis=1)

        print("Serial dedispersion in %f sec" % (time.time()-tbeg))

    # bin down to nfreq_plot freq channels
    full_freq_arr_downsamp = data_dm_max[:nfreq//nfreq_plot*nfreq_plot, :].reshape(\
                                   nfreq_plot, -1, ntime).mean(1)
    # bin down in time by factor of downsamp
    full_freq_arr_downsamp = full_freq_arr_downsamp[:, :ntime//downsamp_res*downsamp_res\
                                   ].reshape(-1, ntime//downsamp_res, downsamp_res).mean(-1)
    
    times = np.linspace(0, ntime_plot*downsamp*dt, len(full_freq_arr_downsamp[0]))

    full_dm_arr_downsamp = full_arr[:, :ntime//downsamp_res*downsamp_res]
    full_dm_arr_downsamp = full_dm_arr_downsamp.reshape(-1, 
                             ntime//downsamp_res, downsamp_res).mean(-1)

    full_freq_arr_downsamp /= np.std(full_freq_arr_downsamp)
    full_dm_arr_downsamp /= np.std(full_dm_arr_downsamp)

    suptitle = " CB:%s  S/N$_{pipe}$:%.1f  S/N$_{presto}$:%.1f\
                 S/N$_{compare}$:%.1f \nDM:%d  t:%.1fs  width:%d" %\
                 (beamno, sig_cut, snr_max, snr_comparison, \
                    dms[dm_max_jj], t0, downsamp)


    if not os.path.isdir('%s/plots' % outdir):
        os.system('mkdir -p %s/plots' % outdir)

    fn_fig_out = '%s/plots/CB%s_snr%d_dm%d_t0%d.pdf' % \
                     (outdir, beamno, sig_cut, dms[dm_max_jj], t0)

    params = sig_cut, dms[dm_max_jj], downsamp, t0, dt
    if mk_plot is True:
        print(fn_fig_out)
        if ndm==1:
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
    fnout = '%s/snr%d_dm%d_t0%d.hdf5'\
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
                        help="make plot if True (default False)", default=False)

    parser.add_option('--multiproc', dest='multiproc', action='store_true', \
                        help="use multicores if True (default False)", default=False)

    parser.add_option('--rficlean', dest='rficlean', action='store_true', \
                        help="use rficlean if True (default False)", default=False)

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

    parser.add_option('--outdir', dest='outdir', type='str',
                        help="directory to write data to", 
                        default='./data/')

    parser.add_option('--compare_trig', dest='compare_trig', type='str',
                        help="Compare input triggers with another trigger file",
                        default=None)

    parser.add_option('--beamno', dest='beamno', type='str',
                        help="Beam number of input data",
                        default='')


    options, args = parser.parse_args()
    fn_fil = args[0]
    fn_sp = args[1]

    if options.save_data == 'concat':
        data_dm_time_full = []
        data_freq_time_full = []
        params_full = []

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
        sig_cut, dm_cut, tt_cut, ds_cut, ind_full = par_1[:, 0], par_1[:, 1], \
                                par_1[:, 2], par_1[:, 3], par_1[:, 4]
    else:
        sig_cut, dm_cut, tt_cut, ds_cut, ind_full = tools.get_triggers(fn_sp, 
                                                         sig_thresh=options.sig_thresh,
                                                         dm_min=options.dm_min,
                                                         dm_max=options.dm_max)

    ntrig_grouped = len(sig_cut)
    print("-----------------------------")
    print("Grouped down to %d triggers" % ntrig_grouped)
    print("----------------------------- \n")

    print("DMs: %s" % dm_cut)
    print("S/N: %s" % sig_cut)

    grouped_triggers = np.empty([ntrig_grouped, 4])
    grouped_triggers[:,0] = sig_cut
    grouped_triggers[:,1] = dm_cut
    grouped_triggers[:,2] = tt_cut
    grouped_triggers[:,3] = ds_cut

    np.savetxt('grouped_pulses.singlepulse', 
                grouped_triggers, fmt='%0.2f %0.1f %0.3f %0.1f')

    for ii, t0 in enumerate(tt_cut[:options.ntrig]):
        try:
            snr_comparison = snr_comparison_arr[ii]
        except:
            snr_comparison=-1

        print("\nStarting DM=%0.2f S/N=%0.2f width=%d time=%f" % (dm_cut[ii], sig_cut[ii], ds_cut[ii], t0))
        data_dm_time, data_freq_time, time_res, params = proc_trigger(\
                                        fn_fil, dm_cut[ii], t0, sig_cut[ii],
                                        mk_plot=options.mk_plot, ndm=options.ndm, 
                                        downsamp=ds_cut[ii], nfreq_plot=options.nfreq_plot,
                                        ntime_plot=options.ntime_plot, cmap=options.cmap,
                                        fn_mask=options.maskfile, cand_no=ii,
                                        multiproc=options.multiproc, 
                                        rficlean=options.rficlean, 
                                        snr_comparison=snr_comparison, 
                                        outdir=options.outdir,
                                        beamno=options.beamno)
        if len(data_dm_time)==0:
            continue

        basedir = options.outdir + '/data/'

        if not os.path.isdir(basedir):
            os.system('mkdir -p %s' % basedir)

        if options.save_data != '0':
            if options.save_data == 'hdf5':
                h5_writer(data_freq_time, data_dm_time, 
                          dm_cut[ii], t0, sig_cut[ii], 
                          beamno='', basedir=basedir, time_res=time_res)
            elif options.save_data == 'npy':
                fnout_freq_time = '%s/data_snr%d_dm%d_t0%f_freq.npy'\
                         % (basedir, sig_cut[ii], dm_cut[ii], np.round(t0, 2))
                fnout_dm_time = '%s/data_snr%d_dm%d_t0%f_dm.npy'\
                         % (basedir, sig_cut[ii], dm_cut[ii], np.round(t0, 2))

                np.save(fnout_freq_time, data_freqtime)
                np.save(fnout_dm_time, data_dmtime)

            elif options.save_data == 'concat':
                data_dm_time_full.append(data_dm_time)
                data_freq_time_full.append(data_freq_time)
                params_full.append(params)
        else:
            print('Not saving data')

    if options.save_data == 'concat':
        data_dm_time_full = np.concatenate(data_dm_time_full)
        data_freq_time_full = np.concatenate(data_freq_time_full)
        
        ndm, nt = data_dm_time.shape
        data_dm_time_full = data_dm_time_full.reshape(-1,ndm,nt)
        nf, nt = data_freq_time.shape
        data_freq_time_full = data_freq_time_full.reshape(-1,nf,nt)
        
        fnout = '%s/data_full.hdf5' % basedir

        f = h5py.File(fnout, 'w')
        f.create_dataset('data_freq_time', data=data_freq_time_full)
        f.create_dataset('data_dm_time', data=data_dm_time_full)
        f.create_dataset('params', data=params_full)
        f.close()

        print('Saved all triggers to %s' % fnout)

    exit()








