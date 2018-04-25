import numpy as np
import scipy.signal

# should there maybe be a clustering class
# and a S/N calculation class?

def dm_range(dm_max, dm_min=5., frac=0.2):
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
        if dm_max < 100.:
            prefac = (1-2*frac)/(1+2*frac)
        if dm_max < 50.:
            prefac = 0.0 

        dm_list.append((int(prefac*dm_max), int(dm_max)))
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

def get_triggers(fn, sig_thresh=5.0, dm_min=0, dm_max=np.inf, t_window=0.5):
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
    if dm_min==0:
        dm_min = 0.9*dm.min()
    if dm_max > 1.1*dm.max():
        dm_max = 1.1*dm.max()

    # Can either do the DM selection here, or after the loop
#    dm_list = dm_range(dm_max, dm_min=dm_min)
    dm_list = dm_range(1.1*dm.max(), dm_min=0.9*dm.min())

    print("Grouping in window of %.2f sec" % np.round(t_window,2))
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

    dm_cut = np.array(dm_cut)
    # now remove the low DM candidates
    ind = np.where((dm_cut >= dm_min) & (dm_cut <= dm_max))[0]

    dm_cut = dm_cut[ind]

    sig_cut = np.array(sig_cut)[ind]
    tt_cut = np.array(tt_cut)[ind]
    ds_cut = np.array(ds_cut)[ind]

    return sig_cut, dm_cut, tt_cut, ds_cut

def calc_snr(data):
    """ Calculate S/N of 1D input array (data)
    after excluding 0.05 at tails
    """
    std_chunk = scipy.signal.detrend(data, type='linear')
    std_chunk.sort()
    ntime_r = len(std_chunk)
    stds = 1.148*np.sqrt((std_chunk[ntime_r//40:-ntime_r//40]**2.0).sum() /
                          (0.95*ntime_r))
    snr_ = std_chunk[-1] / stds 

    return snr_

def calc_snr_widths(data, widths=None):
    """ Calculate the S/N of pulse profile after 
    trying 9 rebinnings.

    Parameters
    ----------
    arr   : np.array
        (ntime,) vector of pulse profile 

    Returns
    -------
    snr : np.float 
        S/N of pulse
    """
    assert len(data.shape)==1
    
    ntime = len(data)
    print(ntime)
    print(data.sum())
    snr_max = 0
    data -= np.median(data)

    if widths is None:
        widths = [1, 2, 4, 8, 16, 32, 64, 128]

#    for ii in range(1, 10):
    for ii in widths:
        for jj in range(ii):
            # skip if boxcar width is greater than 1/4th ntime
            if ii > ntime//8:
                continue
            
            arr_copy = data.copy()
            arr_copy = np.roll(arr_copy, jj)
            arr_ = arr_copy[:ntime//ii*ii].reshape(-1, ii).mean(-1)

            snr_ = calc_snr(arr_)

            if snr_ > snr_max:
                snr_max = snr_
                width_max = ii

    return snr_max, width_max
