import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import glob

from pypulsar.formats import filterbank
from pypulsar.formats import spectra

fn_fil = '/data/01/filterbank/20171010/2017.10.10-02:56:50.B0531+21/CB00.fil'
#fn='/home/arts/leon/scratch/20170306/crab_1hr_dump/liamcrab/crab.liam_1ms_8bit.dfil'
# 'crab4hr_singlepulse.npy'
fn_sp = '/data/01/filterbank/20171010/2017.10.10-02:56:50.B0531+21/CBB_DM56.50.singlepulse'
fdir = '/data/*/filterbank/20171127/2017.11.27-17:24:42.B0329+54/*.fil'

dt = 4.096e-5

def run_prepsubband(fn_fil, lodm, dmstep, numdms, downsamp=1, nsub=128):

    fnout = fn_fil+'.out'
    numout = (30*60./dt)//downsamp*downsamp
    os.system('prepsubband -nsub %d -numout %d -lodm %f \
             -dmstep %f -numdms %d -downsamp %d -o %s %s' % \
             (nsub, numout, lodm, dmstep, numdms, downsamp, fnout, fn_fil))
    return 

def run_single_pulse(fdir):
    flist = glob.glob(fdir+'/*dat')

    for ff in flist:
        print(ff)
        os.system('single_pulse_search.py -t 8.0 %s' % ff)

def dm_range(dm_max, dm_min=2, frac=0.2):

    dm_list =[]
    prefac = (1-frac)/(1+frac)

    while dm_max>dm_min:
        dm_list.append((int(prefac*dm_max), dm_max))
        dm_max = int(prefac*dm_max)
    return dm_list

def get_triggers(fn):
    """ Get brightest trigger in each 10s chunk.
    """
    if fn.split('.')[-1]=='npy':
        A = np.load(fn)
        dm, sig, tt, downs = A[:,0],A[:,1],A[:,2],A[:,4]
    elif fn.split('.')[-1]=='singlepulse':
        A = np.loadtxt(fn)
        dm, sig, tt, downs = A[:,0],A[:,1],A[:,2],A[:,4]
    elif fn.split('.')[-1]=='trigger':
        A = np.loadtxt(fn)
        dm, sig, tt, downs = A[:, -2],A[:, -1],A[:, -3],A[:, 3]
    if len(A)==0:
        return 0, 0, 0, 0
    sig_cut, dm_cut, tt_cut, ds_cut = [],[],[],[]
    
    tduration = tt.max() - tt.min()
    ntime = int(len(tt) / tduration)
    ntime = int(tduration / 1)

    # Make dm windows between 90% of the lowest trigger and 
    # 10% of the largest trigger
    dm_list = dm_range(1.1*dm.max(), dm_min=0.9*dm.min())

    for dms in dm_list:
        for ii in xrange(ntime):
            try:    
                t0, tm = 2*ii, 2*(ii+1)
                ind = np.where((dm<dms[1]) & (dm>dms[0]) & (tt<tm) & (tt>t0))[0]
                sig_cut.append(np.amax(sig[ind]))
                dm_cut.append(dm[ind][np.argmax(sig[ind])])
                tt_cut.append(tt[ind][np.argmax(sig[ind])]) 
                ds_cut.append(downs[ind][np.argmax(sig[ind])])
            except:
                continue

    sig_cut = np.array(sig_cut)
    dm_cut = np.array(dm_cut)
    tt_cut = np.array(tt_cut)
    ds_cut = np.array(ds_cut)
    print(dm_cut)
    print(tt_cut)

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

def proc_trigger(fn_fil, dm0, t0, sig_cut, 
                 ndm=50, mk_plot=False, downsamp=1, beamno='', 
                 fn_mask=None):
    """ Read in filterbank file fn_fil along with 
    dm0 and t0 arrays, save dedispersed data around each 
    trigger. 
    """
    rawdatafile = filterbank.filterbank(fn_fil)

    mask = np.array([  5,   6,   9,  32,  35,  49,  75,  76,  78,  82,  83,  87,  92,
                       93,  97,  98, 108, 110, 111, 112, 114, 118, 122, 123, 124, 157,
                       160, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 660, 661])

    dt = 4.095999975106679e-05
    width = 1.0 # seconds
    freq_up = rawdatafile.header['fch1'] 
    freq_low = freq_up + 1536*rawdatafile.header['foff']
    # Read in 15 disp delays
    width = 5 * abs(4e3 * dm0 * (freq_up**-2 - freq_low**-2))
    print("Using width %f" % width)
    chunksize = int(width/dt)
    start_bin = int((t0 - width/2)/dt)
    start_bin = max(start_bin, 0)

    if start_bin < 0:
        extra = start_bin//2
        start_bin = 0

    dm_min = max(0, dm0-50)
    dm_max = dm0 + 50
    dms = np.linspace(dm_min, dm_max, ndm)
    t_min, t_max = chunksize//2-1500, chunksize//2+1500
    #t_min  = t_min // downsamp
    #t_max = t_max // downsamp
    ntime = t_max-t_min

    full_arr = np.empty([ndm, ntime])   
    dm_max_jj = np.argmin(abs(dms-dm0))

    snr_max = 0
    data = rawdatafile.get_spectra(start_bin, chunksize)
    data.data -= np.median(data.data, axis=-1)[:, None]
    data.data[mask] = 0.

    if not fn_mask is None:
        rfimask = rfifind.rfifind(fn_mask)
        mask = get_mask(rfimask, start_bin, chunksize)
        data = data.masked(mask, maskval='median-mid80')

    for jj, dm_ in enumerate(dms):
        if (dm_<60) and (dm_>50):
            pass
        else:
            continue
        print(ii, dm_)
#        print("Dedispersing to dm=%f starting at t=%d" % (dm_, start_bin))
        #data = rawdatafile.get_spectra(start_bin, chunksize)
        #data.data -= np.median(data.data, axis=-1)[:, None]
        data.dedisperse(dm_)
        dm_arr = data.data[:,t_min:t_max].mean(0)
        snr_ = dm_arr.max() / np.std(dm_arr)
        full_arr[jj] = dm_arr.copy()

        if jj==dm_max_jj:
            data_dm_max = data.data[:, t_min:t_max].copy()

    downsamp = int(3*downsamp)

    # bin down to 32 freq channels
    ddm = data_dm_max[:, :].reshape(32, -1, ntime).mean(1)
    ddm = ddm[:, :ntime//downsamp*downsamp].reshape(-1, ntime//downsamp, downsamp).mean(-1)
    times = np.linspace(0, ntime*dt, len(ddm[0]))
    full_dm_arr_ = full_arr[:, :ntime//downsamp*downsamp].reshape(-1, ntime//downsamp, downsamp).mean(-1)

    if mk_plot is True:

        figure = plt.figure()
        plt.subplot(311)

        ddm /= np.std(ddm)
        plt.imshow(ddm, aspect='auto', vmax=4, vmin=-4, extent=[0, times[-1], freq_up, freq_low])
        plt.ylabel('Freq [MHz]')

        plt.subplot(312)
        plt.plot(times, ddm.mean(0))
        plt.ylabel('Flux')

        plt.subplot(313)
        plt.imshow(full_dm_arr_, aspect='auto', extent=[0, times[-1], dms[-1], dms[0]])
        plt.xlabel('Time [s]')
        plt.ylabel('DM')
    
        plt.suptitle("beam%s snr%d dm%d t0%d" % (beamno, sig_cut, dms[dm_max_jj], tt))

        fn_fig_out = './plots/data_beam%s_snr%d_dm%d_t0%d.pdf' % \
                     (beamno, sig_cut, dms[dm_max_jj], tt)

        plt.savefig(fn_fig_out)
    
    return full_dm_arr_, ddm
#    return full_arr, data_dm_max

if __name__=='__main__':
    import sys

    try:
        mm = int(sys.argv[3])
#        fn_mask = sys.argv[3]
    except IndexError:
        fn_mask = None

    fn_fil = sys.argv[1]
    fn_sp = sys.argv[2]

    sig_cut, dm_cut, tt_cut, ds_cut = get_triggers(fn_sp)

    print("---------------------------")
    print("Grouped down to %d triggers" % len(sig_cut))
    print("--------------------------- \n")

    for ii, tt in enumerate(tt_cut[100*mm:100*(mm+1)]):
        
        print(ii,dm_cut[ii])
        data_dmtime, data_freqtime = proc_trigger(fn_fil, 56.8, 11.9706, 30, mk_plot=True, ndm=100, downsamp=ds_cut[ii])
        #data_dmtime, data_freqtime = proc_trigger(fn_fil, dm_cut[ii], tt, sig_cut[ii], \
        #                                          mk_plot=True, ndm=100, downsamp=ds_cut[ii])

        fnout_freqtime = './data_trainsnr%d_dm%d_t0%d_freq.npy' % (sig_cut[ii], dm_cut[ii], tt)
        fnout_dmtime = './data_trainsnr%d_dm%d_t0%d_dm.npy' % (sig_cut[ii], dm_cut[ii], tt)

        np.save(fnout_freqtime, data_freqtime)
        np.save(fnout_dmtime, data_dmtime)

    exit()


#     flist = glob.glob('./all*singlepulse')
# #    mm=int(sys.argv[1])
#     for ff in flist[:]:
#         sig_cut, dm_cut, tt_cut, ds_cut = get_triggers(ff)

#         beamno = ff.split('_')[1][:2]
#         print(beamno)

#         fn_fil = glob.glob('/data/*/filterbank/20171127/2017.11.27-17:24:42.B0329+54/CB%s.fil' % beamno)[0]
#         print(fn_fil)
#         try:
#             print("%s with %d triggers" % (ff, len(sig_cut)))
#         except:
#             print("Skipping %s" % beamno)
#             continue 

#         for ii, tt in enumerate(tt_cut):
#             print(ii, tt, sig_cut[ii], ds_cut[ii])
#             data_dmtime, data_freqtime = proc_trigger(fn_fil, dm_cut[ii], tt, sig_cut[ii], \
#                                                       mk_plot=True, ndm=25, downsamp=ds_cut[ii],\
#                                                       beamno=beamno)
            
#             fnout_freqtime = './data%s_snr%d_dm%d_t0%d_freq.npy' % (beamno, sig_cut[ii], dm_cut[ii], tt)
#             fnout_dmtime = './data%s_snr%d_dm%d_t0%d_dm.npy' % (beamno, sig_cut[ii], dm_cut[ii], tt)

#             np.save(fnout_freqtime, data_freqtime)
#             np.save(fnout_dmtime, data_dmtime)


#     mm = 10
#     flist = glob.glob(fdir)

#     for fn_fil in flist:
#         fdir = fn_fil.split('CB')[0]
#         beamno = fn_fil.split('CB')[-1][:2]
#         os.system('cat %s*%s*%s > all_%s.singlepulse' % (fdir, beamno, 'singlepulse', beamno))

#     exit()

#     flist = flist[mm*3:(mm+1)*3]

#     for fn_fil in flist:
#         fdir = fn_fil.split('CB')[0]
#         print(fn_fil)
#         print(fdir)
#         run_prepsubband(fn_fil, 20.00, 0.3, 50, downsamp=1, nsub=128)
#         run_single_pulse(fdir)

#     exit()

#     sig_cut, dm_cut, tt_cut = get_triggers(fn_sp)


# fn_fil = '/data/09/filterbank/20171127/2017.11.27-17:24:42.B0329+54/CB21.fil'
# fn_sp = 'all_21.singlepulse'










