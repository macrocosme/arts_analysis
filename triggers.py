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
#sig_cut, dm_cut, tt_cut = get_triggers('crab4hr_singlepulse.npy')
fn_sp = '/data/01/filterbank/20171010/2017.10.10-02:56:50.B0531+21/CBB_DM56.50.singlepulse'
fdir = '/data/*/filterbank/20171127/2017.11.27-17:24:42.B0329+54/*.fil'

dt = 4.096e-5

def run_prepsubband(fn_fil, lodm, dmstep, numdms, downsamp=1, nsub=128):

    fnout = fn_fil+'.out'
    print(fnout)
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

def get_triggers2(fn):
    """ Get brightest trigger in each 10s chunk.
    """
    if fn.split('.')[-1]=='npy':
        A = np.load(fn)
    elif fn.split('.')[-1]=='singlepulse':
        A = np.loadtxt(fn)

    dm, sig, tt, downs = A[:,0],A[:,1],A[:,2],A[:,4]
    sig_cut, dm_cut, tt_cut = [],[],[]
    
    for ii in xrange(10*4*3600//10):
        t0, tm = 10*ii, 10*(ii+1)                                                                 
        ind = np.where((tt<tm) & (tt>t0))[0]
        try: 
            sig_cut.append(sig[ind].max())
            dm_cut.append(dm[ind][np.argmax(sig[ind])])
            tt_cut.append(tt[ind][np.argmax(sig[ind])])        
        except:                                     
            continue

    sig_cut = np.array(sig_cut)
    dm_cut = np.array(dm_cut)
    tt_cut = np.array(tt_cut)

    return sig_cut, dm_cut, tt_cut

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
    elif fn.split('.')[-1]=='singlepulse':
        A = np.loadtxt(fn)

    if len(A)==0:
        return 0, 0, 0, 0

    dm, sig, tt, downs = A[:,0], A[:,1], A[:,2], A[:,4]
    sig_cut, dm_cut, tt_cut, ds_cut = [],[],[], []
    
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
                sig_cut.append(np.argmax(sig[ind]))
                dm_cut.append(dm[ind][np.argmax(sig[ind])])
                tt_cut.append(tt[ind][np.argmax(sig[ind])]) 
                ds_cut.append(downs[ind][np.argmax(sig[ind])])
            except:
                continue

    sig_cut = np.array(sig_cut)
    dm_cut = np.array(dm_cut)
    tt_cut = np.array(tt_cut)
    ds_cut = np.array(ds_cut)

    return sig_cut, dm_cut, tt_cut, ds_cut

def proc_trigger(fn_fil, dm0, t0, sig_cut, 
                 ndm=50, mk_plot=False, downsamp=1, beamno=''):
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
    # Read in three disp delays
    width = 15 * abs(4e3 * dm0 * (freq_up**-2 - freq_low**-2))
    print("Using width %f" % width)
    chunksize = int(width/dt)
    start_bin = int((t0 - width/2)/dt)
    start_bin = max(start_bin, 0)

    if start_bin < 0:
        extra = start_bin//2
        start_bin = 0

    dm_min = max(0, dm0-10)
    dm_max = dm0 + 10
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
#    data.data[mask] = 0.

    for jj, dm_ in enumerate(dms):
        print("Dedispersing to dm=%f starting at t=%d" % (dm_, start_bin))
        #data = rawdatafile.get_spectra(start_bin, chunksize)
        #data.data -= np.median(data.data, axis=-1)[:, None]
        data.dedisperse(dm_)
        dm_arr = data.data[:,t_min:t_max].mean(0)
        snr_ = dm_arr.max() / np.std(dm_arr)
        full_arr[jj] = dm_arr.copy()

        if jj==dm_max_jj:
            data_dm_max = data.data[:, t_min:t_max].copy()

    if mk_plot is True:

        downsamp = int(downsamp)

        figure = plt.figure()
        plt.subplot(311)
        ddm = data_dm_max[:, :].reshape(-1, 16, ntime).mean(1)
        ddm = ddm[:, :ntime//downsamp*downsamp].reshape(-1, ntime//downsamp, downsamp).mean(-1)

        ddm /= np.std(ddm)
        plt.imshow(ddm, aspect='auto', vmax=4, vmin=-4, extent=[0, ntime*dt*downsamp, freq_up, freq_low])
        plt.ylabel('Freq [MHz]')

        plt.subplot(312)
        plt.plot(ddm.mean(0))
        plt.ylabel('Flux')

        plt.subplot(313)
        full_dm_arr_ = full_arr[:, :ntime//downsamp*downsamp].reshape(-1, ntime//downsamp, downsamp).mean(-1)
        plt.imshow(full_dm_arr_, aspect='auto', extent=[0, ntime*dt*downsamp, dms[0], dms[1]])
        plt.xlabel('Time')
        plt.ylabel('DM')
    
        plt.subtitle("data_beam%s_snr%d_dm%d_t0%d" % (beamno, sig_cut, dms[dm_max_jj], tt))

        fn_fig_out = './plots/data_beam%s_snr%d_dm%d_t0%d.pdf' % \
                     (beamno, sig_cut, dms[dm_max_jj], tt)

        plt.savefig(fn_fig_out)

    return full_arr, data_dm_max

if __name__=='__main__':
    import sys

    fn_fil = sys.argv[1]
    fn_sp = sys.argv[2]

    sig_cut, dm_cut, tt_cut, ds_cut = get_triggers(fn_sp)

    for ii, tt in enumerate(tt_cut):
        data_dmtime, data_freqtime = proc_trigger(fn_fil, dm_cut[ii], tt, sig_cut[ii], \
                                                  mk_plot=True, ndm=25, downsamp=ds_cut[ii])

        fnout_freqtime = './data_snr%d_dm%d_t0%d_freq.npy' % (sig_cut[ii], dm_cut[ii], tt)
        fnout_dmtime = './data_snr%d_dm%d_t0%d_dm.npy' % (sig_cut[ii], dm_cut[ii], tt)

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










