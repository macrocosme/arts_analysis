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
                   cand_no=1, fnout='out.pdf', suptitle=''):
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
    ax2.plot(DM0_delays, freqs, c='r', lw='2', alpha=0.5)
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

    suptitle = 'p:%.2f %s' % (prob, suptitle)

#    try:
#        fig.suptitle("p: {:.2f}, S/N: {:.0f}, DM: {:.2f}, \
#                  T0: {:.2f}, CB: {:02d}".format(prob, snr, dm, t0, cb))
#        figname = "plots/cand_{:04d}_snr{:.0f}_dm{:.0f}.pdf".format(cand_no, snr, dm)
#    except:
#        fig.suptitle("p: %.2f, S/N: %.0f, DM: %.2f, T0: %.2f, CB: %02d" \
#                     % (prob, snr, dm, t0, cb))
#        figname = "plots/cand_%04d_snr%.0f_dm%.0f.pdf" % (cand_no, snr, dm)

    fig.suptitle(suptitle)
    plt.savefig(fnout)
    plt.close(fig)

def plot_three_panel(data_freq_time, data_dm_time, params, dms, times=None, 
                     freq_up=1549.90234375, freq_low=1250.09765625,
                     cmap="RdBu", suptitle="", fnout="out.pdf", cb=None,
                     cand_no=1, prob=None):
    """ Plot freq/time, time, and dm/time data in 
    three panels. 
    """
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
               extent=[0, times[-1], dms[-1], dms[0]], 
               interpolation='nearest', cmap=cmap, origin='upper')
    plt.xlabel('Time [ms]')
    plt.ylabel('DM', labelpad=10)

    DM0_delays = dm * 4.15E6 * (freq_low**-2 - freqs**-2)
    ax1.plot(DM0_delays, freqs, c='r', lw='2', alpha=0.5)
    plt.xlim(0, times[-1])

    if cb is None:
        cb = -1
    if prob is None:
        prob = -1

    suptitle = 'p:%.2f %s' % (prob, suptitle)

    plt.suptitle(suptitle, fontsize=14)
#    plt.tight_layout()
    plt.savefig(fnout)

def plot_from_h5(fn, cb, freq_low=1250.09765625, freq_up=1549.90234375, 
                 ):
    # read dataset 
    with h5py.File(fn, 'r') as f:
        data_frb_candidate = f['data_frb_candidate'][:]
        probability = f['probability'][:]
        params = f['params'][:]  # snr, DM, boxcar width, arrival time

    for i, cand in enumerate(data_frb_candidate):
        data_freq_time = cand[:, :, 0]

        plot_two_panel(data_freq_time, params[i], cb=cb, freq_low=freq_low, 
                    freq_up=freq_up, prob=probability[i], cand_no=i)


def mk_histograms(params, fnout='summary_hist.pdf', 
                  suptitle='Trigger summary', alpha=0.25):
    """ Take parameter array (ntrig, 4) and make 
    histograms of each param 
    """
    assert(len(params)==4)

    # if int(mpl.__version__[0])<2:
    #     alpha=0.5
    # else:
    #     alpha=1.

    figure = plt.figure(figsize=(8,8))
    ax1 = plt.subplot(221)
    plt.hist(params[0], log=True, color='C0', alpha=alpha, bins=30)
    plt.xlabel('DM [pc cm**-3]', fontsize=12)

    plt.subplot(222)
    plt.hist(params[1], color='C1', alpha=alpha, log=True, bins=30)
    plt.xlabel('S/N', fontsize=12)

    plt.subplot(223)
    plt.hist(params[2], color='C2', alpha=alpha, log=True, bins=30)
    plt.xlabel('Time [sec]', fontsize=12)

    plt.subplot(224)
    plt.hist(np.log2(params[3]), color='C3', alpha=alpha, bins=8, log=True)
    plt.xlabel('log2(Width) [samples]', fontsize=12)

    suptitle = "%s %d events" % (suptitle, len(params))
    plt.suptitle(suptitle)

    plt.show()
    plt.savefig(fnout)

def plot_comparison(par_1, par_2, par_match_arr, 
                    ind_missed, figname='./test.pdf', 
                    suptitle='file 2 vs. file 1.'):
    fig = plt.figure(figsize=(12,14))

    frac_recovered = len(ind_missed)

    snr_1, snr_2 = par_1[0], par_2[0]
    dm_1, dm_2 = par_1[1], par_2[1]
    t_1, t_2 = par_1[2], par_2[2]
    width_1, width_2 = par_1[3], par_2[3]

    snr_1_match = par_match_arr[0,:,0]
    snr_2_match = par_match_arr[0,:,1]

    dm_1_match = par_match_arr[1,:,0]
    dm_2_match = par_match_arr[1,:,1]

    width_1_match = par_match_arr[3,:,0]
    width_2_match = par_match_arr[3,:,1]

    fig.add_subplot(331)
    plt.plot(snr_1[ind_missed], 5+np.zeros([len(ind_missed)]), '.', color='orange')
    plt.plot(snr_1_match, snr_2_match, '.')
    plt.plot(snr_1, snr_1, color='k')
    plt.loglog()
    plt.xlabel('file 1 S/N', fontsize=12)
    plt.ylabel('file 2 S/N', fontsize=12)        
    plt.legend(['Missed', 'Matched', 'Equal S/N'], fontsize=10)

    fig.add_subplot(334)
    plt.plot(dm_1[ind_missed], np.zeros([len(ind_missed)]), '.', color='orange')
    plt.plot(dm_1_match, snr_1_match/snr_2_match, '.')
    plt.plot(dm_1, np.ones_like(dm_1), '--', color='k')
    plt.xlabel('DM', fontsize=12)
    plt.ylabel('S/N_1 : S/N_2', fontsize=12)        
    plt.legend(['Missed','Detected','Equal S/N'], fontsize=10)

    plt.subplot(332)
    plt.hist(dm_1, log=True, alpha=0.5, bins=30)
    plt.hist(dm_2, log=True, alpha=0.5, bins=30)
    plt.xlabel('DM [pc cm**-3]', fontsize=12)
    plt.legend(['file 1','file 2'], fontsize=12)

    plt.subplot(333)
    plt.hist(snr_1, alpha=0.5, log=True, bins=30)
    plt.hist(snr_2, alpha=0.5, log=True, bins=30)
    plt.xlabel('S/N', fontsize=12)

    plt.subplot(335)
    plt.hist(t_1, alpha=0.5, log=True, bins=30)
    plt.hist(t_2, alpha=0.5, log=True, bins=30)
    plt.xlabel('Time [sec]', fontsize=12)

    plt.subplot(336)
    plt.hist(np.log2(width_1), alpha=0.5, bins=8, log=True)
    plt.hist(np.log2(width_2), alpha=0.5, bins=8, log=True)
    plt.xlabel('log2(Width) [samples]', fontsize=12)

    # fig.add_subplot(337)
    # plt.hist(width_1[ind_missed], bins=50, alpha=0.3, normed=True, color='C3')
    # plt.hist(width_1, bins=50, alpha=0.3, normed=True)
    # plt.hist(width_2, bins=50, alpha=0.3, normed=True)
    # plt.xlabel('Width [samples]', fontsize=12)

    fig.add_subplot(325)
    plt.plot(np.log2(width_1), np.log10(snr_1), '.', color='C3')
    plt.plot(np.log2(width_1_match), np.log10(snr_2_match),'.')
#    plt.plot(width_1_match, snr_2_match,'.')
    plt.xlabel('log2(Width) [samples]', fontsize=12)
    plt.ylabel('S/N injected', fontsize=12)
    plt.legend(['file 1 all', 'Matched'], fontsize=10)
    plt.grid()

    fig.add_subplot(326)
    plt.plot(np.log2(width_1), np.log10(0.1+dm_1),'.',color='C3')
    plt.plot(np.log2(width_1_match), np.log10(0.1+dm_2_match),'.')
    #plt.plot(np.log2(width_1_match), np.log10(0.1+dm_2_match),'.')
    plt.xlabel('Width [samples]', fontsize=12)
    plt.ylabel('DM', fontsize=12)
    plt.legend(['file 1 all', 'Matched'], fontsize=10)
    plt.grid()

    snr_ratio = np.mean(snr_1_match / snr_2_match)

    suptitle += ('   avg (S/N)_1/(S/N)_2: ' + np.str(np.round(snr_ratio,2)))

    plt.suptitle(suptitle, fontsize=20)
    #plt.tight_layout()
    plt.show()
    plt.savefig(figname)

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


