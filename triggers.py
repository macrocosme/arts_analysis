import numpy as np

from pypulsar.formats import filterbank
from pypulsar.formats import spectra

fn='/data/01/filterbank/20171010/2017.10.10-02:56:50.B0531+21/CB00.fil'

dt = 4.095999975106679e-05
start_bin = 10000000+75000
chunksize = int(1.0 / dt)
downsamp = 1
full_arr = np.empty([100, chunksize])

def get_triggers(fn):
    """ Get brightest trigger in each 10s chunk.
    """
    A = np.load(fn)
    dm, sig, tt, downs = A[:,0],A[:,1],A[:,2],A[:,4]
    sig_cut, dm_cut, ttc = [],[],[]

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

def func(dm0, t0, ndm=50):
    dt = 4.095999975106679e-05
    width = 1.0 # seconds
    chunksize = int(width/dt)
    start_bin = int((t0 - width/2)/dt)

    dm_min = max(0, dm0-5)
    dm_max = dm0+5
    dms = np.linspace(dm_min, dm_max, ndm)
    t_min, t_max = chunksize//2-200, chunksize//2+200

    full_arr = np.empty([ndm, 400])   
    dm_max_jj = np.argmin(abs(dms-dm0))

    for jj, dm_ in enumerate(dms):
        print(dm_)
        data = rawdatafile.get_spectra(start_bin, chunksize)
        data.downsample(downsamp)
        data.data -= np.median(data.data, axis=-1)[:, None]
        data.dedisperse(dm_)
        dm_arr = data.data[:,t_min:t_max].mean(0)
        full_arr[jj] = dm_arr

        if jj==dm_max_jj:
            np.save('data', data)
            data_dm_max = data.data[:, t_min:t_max].copy()

    return full_arr, data_dm_max


sig_cut, dm_cut, tt_cut = get_triggers('crab4hr_singlepulse.npy')

for ii, tt in enumerate(tt_cut):
    print(ii, tt)
    a, d = func(dm_cut[ii], tt)











