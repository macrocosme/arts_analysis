import numpy as np
import h5py, glob
import optparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
    

def mk_histograms(params, fnout='summary_hist.pdf', 
                  suptitle='Trigger summary'):
    """ Take parameter array (ntrig, 4) and make 
    histograms of each param 
    """
    assert(params.shape[-1]==4)

    if int(mpl.__version__[0])<2:
        alpha=0.5
    else:
        alpha=1.

    figure = plt.figure(figsize=(8,8))
    ax1 = plt.subplot(221)
    plt.hist(np.log10(params[:,0]), log=True, color='C0', alpha=alpha)
    plt.xlabel('log(S/N)', fontsize=12)

    plt.subplot(222)
    plt.hist(params[:,1], color='C1', alpha=alpha)
    plt.xlabel('DM pc cm**-3', fontsize=12)

    plt.subplot(223)
    plt.hist(params[:,2], color='C2', alpha=alpha)
    plt.xlabel('time resolution [sec]', fontsize=12)

    plt.subplot(224)
    plt.hist(params[:,3], color='C3', alpha=alpha)
    plt.xlabel('Arrival time [sec]', fontsize=12)

    suptitle = "%s %d events" % (suptitle, len(params))
    plt.suptitle(suptitle)

    plt.show()
    plt.savefig(fnout)


def file_reader(fn, ftype='hdf5'):
    if ftype is 'hdf5':
        f = h5py.File(fn, 'r')

        data_freq_time = f['data_freq_time'][:]

        try:
            data_dm_time = f['data_dm_time'][:]
        except:
            data_dm_time = 0

        attr = f.attrs.items()

        snr, dm0, time_res, t0 = attr[0][1], attr[1][1], attr[5][1], attr[6][1] 

        f.close()

        return data_freq_time, data_dm_time, [snr, dm0, time_res, t0]

    elif ftype is 'npy':
        data = np.load(fn)

        return data

def concat_files(fdir, ftype='hdf5', nfreq_f=32, 
                 ntime_f=64, fnout='./'):
    """ Read in a list of files, extract central ntime_f 
    times, bin down to nfreq_f, and write to a single h5 
    file. 
    """

    file_list = glob.glob('%s/*%s' % (fdir, ftype))

    data_freq_time_full, data_dm_time_full, params_full = [], [], []
    ntrig = len(file_list)

    assert ntrig!=0, "Empty file list"
    print("Processing %d files" % ntrig)
    print('Dir: %s*%s' % (fdir, ftype))

    for fn in file_list[:]:
        if 'data_train_full' in fn:
            continue

        data_freq_time, data_dm_time, params = file_reader(fn, ftype=ftype)

        nfreq, ntime = data_freq_time.shape
        params_full.append(params)

        if ntime != ntime_f:
            tl, th = ntime//2-ntime_f//2, ntime//2+ntime_f//2
            data_freq_time = data_freq_time[:, tl:th]

        if nfreq > nfreq_f:
            data_freq_time = data_freq_time[:nfreq//nfreq_f*nfreq_f]
            data_freq_time = data_freq_time.reshape(nfreq_f, -1, ntime_f).mean(1)

        #normalize to zero median, unit variaance
        data_freq_time -= np.median(data_freq_time)
        data_freq_time /= np.std(data_freq_time)

        try:
            ndm = data_dm_time.shape[0]
            data_dm_time = data_dm_time[:, tl:th]
            
            #normalize to zero median, unit variance
            data_dm_time -= np.median(data_dm_time)
            data_dm_time /= np.std(data_dm_time)

            data_dm_time_full.append(data_dm_time)
        except:
            pass 

        data_freq_time_full.append(data_freq_time)

    print("Writing out to %s" % fnout)
    g = h5py.File(fnout, 'w')

    try:
        data_dm_time_full = np.concatenate(data_dm_time_full)
        data_dm_time_full = data_freq_time_full.reshape(-1, ndm, ntime_f)
        
        g.create_dataset('data_dm_time', data=data_dm_time_full)
    except:
        pass 

    params_full = np.concatenate(params_full)
    params_full = params_full.reshape(-1, 4)

    data_freq_time_full = np.concatenate(data_freq_time_full)
    data_freq_time_full = data_freq_time_full.reshape(-1, nfreq_f, ntime_f)

    g.create_dataset('data_freq_time', data=data_freq_time_full)
    g.create_dataset('params', data=params_full)
    # Indicate that data has no labels
    y = -1*np.ones([ntrig])
    g.create_dataset('labels', data=y)
    g.close()

    return data_freq_time_full, data_dm_time_full, params_full


if __name__=='__main__':
    parser = optparse.OptionParser(prog="preprocess.py", \
                        description="Create a waterfall plot to show the " \
                                    "frequency sweep of a single pulse " \
                                    "in psrFits data.")
    parser.add_option('--fnout', dest='fnout', type='str', \
                      help="Name of hdf5 file to save data to", \
                      default='./all_data.hdf5')

    parser.add_option('--nfreq_f', dest='nfreq_f', type='int', \
                        help="Number of frequency channels to keep", 
                        default=32)
    parser.add_option('--ntime_f', dest='ntime_f', type='int', \
                        help="Number of time samples to store around central pixel", 
                     default=64)

    parser.add_option('--mk_plot', dest='mk_plot', type='int', \
                        help="Make histogram summary plot", default=1)

    options, args = parser.parse_args()
    fdir = args[0]

    data_freq_time, data_dm_time, params = concat_files(fdir, 
                                            ftype='hdf5', 
                                            nfreq_f=options.nfreq_f, 
                                            ntime_f=options.ntime_f, 
                                            fnout=options.fnout)

    if options.mk_plot:
        mk_histograms(params, fnout='summary_hist.pdf', 
                  suptitle='Trigger summary')





