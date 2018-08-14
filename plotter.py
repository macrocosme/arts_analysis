#!/usr/bin/env python
#
# Plot triggers output by the ML classifier

import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

def plot_two_panel(data_freq_time, params, times=None, cb=None, prob=None, 
                   freq_low=1250.09765625, freq_up=1549.90234375, 
                   cand_no=1):
    """ Plot data in two panels
    """
    snr, dm, bin_width, t0, delta_t = params
    nfreq, ntime = data_freq_time.shape

    if times is None:
        times = np.arange(ntime)*delta_t*bin_width*1e3  # ms

    times *= 1e3 #convert to ms
    freqs = np.linspace(freq_low, freq_up, nfreq)

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, 
                                gridspec_kw=dict(height_ratios=[1, 2]))

    # timeseries
    ax1.plot(times, np.sum(data_freq_time, axis=0)/
            np.sqrt(data_freq_time.shape[0]), c='k')
    ax1.set_ylabel('S/N', labelpad=10)
    # add what a DM=0 signal would look like
    DM0_delays = dm * 4.15E6 * (freq_low**-2 - freqs**-2)
    ax2.plot(DM0_delays, freqs, c='r', lw='2')
    print(DM0_delays, freqs)
    # scaling: std = 1, median=0
    extent = [times[0], times[-1], freq_low, freq_up]

    ax2.imshow(data_freq_time, cmap='viridis', vmin=-3, vmax=3, 
               interpolation='nearest', aspect='auto', 
               origin='upper', extent=extent)

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Freq (MHz)', labelpad=10)

    if cb is None:
        cb = -1
    if prob is None:
        prob = -1

    try:
        fig.suptitle("p: {:.2f}, S/N: {:.0f}, DM: {:.2f}, \
                  T0: {:.2f}, CB: {:02d}".format(prob, snr, dm, t0, cb))
        figname = "plots/cand_{:04d}_snr{:.0f}_dm{:.0f}.pdf".format(cand_no, snr, dm)
    except:
        fig.suptitle("p: %.2f, S/N: %.0f, DM: %.2f, T0: %.2f, CB: %02d" \
                     % (prob, snr, dm, t0, cb))
        figname = "plots/cand_%04d_snr%.0f_dm%.0f.pdf" % (cand_no, snr, dm)

    plt.savefig(figname)
    plt.close(fig)

def plot_three_panel(data_freq_time, data_dm_time, params, dms, times=None, 
                     freq_up=1549.90234375, freq_low=1250.09765625,
                     cmap="RdBu", suptitle="", fnout="out.pdf", 
                     cand_no=1):
    snr, dm, bin_width, t0, delta_t = params
    nfreq, ntime = data_freq_time.shape
    freqs = np.linspace(freq_low, freq_up, nfreq)

    if times is None:
        times = np.arange(ntime)*delta_t*bin_width*1e3  # ms

    figure = plt.figure()
    ax1 = plt.subplot(311)

    times *= 1e3 # convert to ms from s

    plt.imshow(data_freq_time, aspect='auto', vmax=4, vmin=-4, 
               extent=[0, times[-1], freq_low, freq_up], 
               interpolation='nearest', cmap=cmap)
    plt.ylabel('Freq [MHz]', labelpad=10)

    plt.subplot(312, sharex=ax1)
    plt.plot(times, data_freq_time.mean(0), color='k')
    plt.ylabel('Flux', labelpad=10)

    plt.subplot(313, sharex=ax1)
    plt.imshow(data_dm_time, aspect='auto', 
               extent=[0, times[-1], dms[0], dms[-1]], 
               interpolation='nearest', cmap=cmap)
    plt.xlabel('Time [ms]')
    plt.ylabel('DM', labelpad=10)

    DM0_delays = dm * 4.15E6 * (freq_low**-2 - freqs**-2)
    ax1.plot(DM0_delays, freqs, c='r', lw='2')

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()
    plt.savefig(fnout)

def plot_from_h5(fn, cb, freq_low=1250.09765625, freq_up=1549.90234375, 
                 ):
    # read dataset 
    print(freq_low, freq_up)
    with h5py.File(fn, 'r') as f:
        data_frb_candidate = f['data_frb_candidate'][:]
        probability = f['probability'][:]
        params = f['params'][:]  # snr, DM, boxcar width, arrival time

    for i, cand in enumerate(data_frb_candidate):
        data_freq_time = cand[:, :, 0]

        plot_two_panel(data_freq_time, params[i], cb=cb, freq_low=freq_low, 
                    freq_up=freq_up, prob=probability[i], cand_no=i)


if __name__ == '__main__':
#     # input hdf5 file
    print('\nExpecting: data_file CB <freq_low> <freq_up>\n')
    fn = sys.argv[1]
    cb = int(sys.argv[2])

    try:
        freq_low = np.float(sys.argv[3])
    except:
        freq_low = 1250.09765625

    try:
        freq_up = np.float(sys.argv[4])
    except:
        freq_up = 1549.90234375
        
    plot_from_h5(fn, cb, freq_low=freq_low, freq_up=freq_up)


