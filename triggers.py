import numpy as np

from pypulsar.formats import filterbank
from pypulsar.formats import spectra

import matplotlib.pyplot as plt

fn_fil = '/data/01/filterbank/20171010/2017.10.10-02:56:50.B0531+21/CB00.fil'
#fn='/home/arts/leon/scratch/20170306/crab_1hr_dump/liamcrab/crab.liam_1ms_8bit.dfil'
#sig_cut, dm_cut, tt_cut = get_triggers('crab4hr_singlepulse.npy')
fn_sp = '/data/01/filterbank/20171010/2017.10.10-02:56:50.B0531+21/CBB_DM56.50.singlepulse'

dt = 4.095999975106679e-05

def get_triggers(fn):
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

def proc_trigger(fn_fil, dm0, t0, ndm=50, mk_plot=False, downsamp=1):
    """ Read in filterbank file fn_fil along with 
    dm0 and t0 arrays, save dedispersed data around each 
    trigger. 
    """
    rawdatafile = filterbank.filterbank(fn_fil)

    dt = 4.095999975106679e-05
    width = 1.0 # seconds
    freq_up = rawdatafile.header['fch1'] 
    freq_low = freq_up + 1536*rawdatafile.header['foff']
    # Read in three disp delays
    width = 3 * abs(4e3 * dm0 * (freq_up**-2 - freq_low**-2))
    print("Using width %f" % width)
    chunksize = int(width/dt)
    start_bin = int((t0 - width/2)/dt)

    if start_bin < 0:
        extra = start_bin//2
        start_bin = 0

    dm_min = max(0, dm0-5)
    dm_max = dm0+5
    dms = np.linspace(dm_min, dm_max, ndm)
    t_min, t_max = chunksize//2-200, chunksize//2+200

    full_arr = np.empty([ndm, 400])   
    dm_max_jj = np.argmin(abs(dms-dm0))

    snr_max = 0

    for jj, dm_ in enumerate(dms):
        print("Dedispersing to dm=%f starting at t=%d" % (dm_, start_bin))
        data = rawdatafile.get_spectra(start_bin, chunksize)

        if downsamp > 1:
            data.downsample(downsamp)

        data.data -= np.median(data.data, axis=-1)[:, None]
        data.dedisperse(dm_)
        dm_arr = data.data[:,t_min:t_max].mean(0)
        snr_ = dm_arr.max() / np.std(dm_arr)
        full_arr[jj] = dm_arr

        if jj==dm_max_jj:
#        if snr_>=snr_max:
            data_dm_max = data.data[:, t_min:t_max].copy()

    if mk_plot is True:
        figure = plt.figure()
        plt.subplot(311)
        ddm = data_dm_max[:, 100:300].reshape(-1, 16, 200).mean(1)
        plt.imshow(ddm, aspect='auto')

        plt.subplot(312)
        plt.plot(data_dm_max.mean(0))

        plt.subplot(313)
        plt.imshow(full_arr, aspect='auto')
        
        fn_fig_out = './data_snr%d_dm%d_t0%d.pdf' % \
                     (sig_cut[ii], dm_cut[ii], tt)

        plt.savefig(fn_fig_out)

    return full_arr, data_dm_max


sig_cut, dm_cut, tt_cut = get_triggers(fn_sp)

print("Using %d events" % len(sig_cut))

for ii, tt in enumerate(tt_cut):
    print(ii, tt, sig_cut[ii])
    data_dmtime, data_freqtime = proc_trigger(fn_fil, dm_cut[ii], tt, mk_plot=True, ndm=25)

    fnout_freqtime = './data_snr%d_dm%d_t0%d_freq.npy' % (sig_cut[ii], dm_cut[ii], tt)
    fnout_dmtime = './data_snr%d_dm%d_t0%d_dm.npy' % (sig_cut[ii], dm_cut[ii], tt)

    np.save(fnout_freqtime, data_freqtime)
    np.save(fnout_dmtime, data_dmtime)










