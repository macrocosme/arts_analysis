import numpy as np
import scipy.signal

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

        # skip if boxcar width is greater than 1/4th ntime
        if ii > ntime//8:
            continue
            
        arr_copy = data.copy()
        arr_ = arr_copy[:ntime//ii*ii].reshape(-1, ii).mean(-1)

        snr_ = calc_snr(arr_)

        print(ii, snr_)

        if snr_ > snr_max:
            snr_max = snr_
            width_max = ii

    return snr_max



