import time

import random
import numpy as np
import glob
from scipy import signal
import optparse

try:
    import matplotlib.pyplot as plt
except:
    plt = None
    pass

import simulate_frb
import reader
#import rfi_test

def inject_in_filterbank_background(fn_fil):
    """ Inject an FRB in each chunk of data 
        at random times. Default params are for Apertif data.
    """

    chunksize = 5e5
    ii=0

    data_full =[]
    nchunks = 250
    nfrb_chunk = 8
    chunksize = 2**16

    for ii in range(nchunks):
        downsamp = 2**((np.random.rand(nfrb_chunk)*6).astype(int))

        try:
            # drop FRB in random location in data chunk
            rawdatafile = filterbank.filterbank(fn_fil)
            dt = rawdatafile.header['tsamp']
            freq_up = rawdatafile.header['fch1']
            nfreq = rawdatafile.header['nchans']
            freq_low = freq_up + nfreq*rawdatafile.header['foff']
            data = rawdatafile.get_spectra(ii*chunksize, chunksize)
        except:
            continue
    

        #dms = np.random.uniform(50, 750, nfrb_chunk)
        dm0 = np.random.uniform(90, 750)
        end_width = abs(4e3 * dm0 * (freq_up**-2 - freq_low**-2))
        data.dedisperse(dm0)
        NFREQ, NT = data.data.shape

        print("Chunk %d with DM=%.1f" % (ii, dm0))
        for jj in xrange(nfrb_chunk):
            if 8192*(jj+1) > (NT - end_width):
                print("Skipping at ", 8192*(jj+1))
                continue
            data_event = data.data[:, jj*8192:(jj+1)*8192]
            data_event = data_event.reshape(NFREQ, -1, downsamp[jj]).mean(-1)
            print(data_event.shape)
            data_event = data_event.reshape(32, 48, -1).mean(1)

            NTIME = data_event.shape[-1]
            data_event = data_event[..., NTIME//2-125:NTIME//2+125]
            data_event -= np.mean(data_event, axis=-1, keepdims=True)
            data_full.append(data_event)

    data_full = np.concatenate(data_full)
    data_full = data_full.reshape(-1, 32, 250)

    np.save('data_250.npy', data_full)


def inject_in_filterbank(fn_fil, fn_out_dir, N_FRBs=1, 
                         NFREQ=1536, NTIME=2**15, rfi_clean=False,
                         dm=2500.0, freq=(1250, 1550), dt=0.00004096,
                         chunksize=5e4):
    """ Inject an FRB in each chunk of data 
        at random times. Default params are for Apertif data.
    """

    if type(dm) is not tuple:
        max_dm = dm
    else:
        max_dm = max(dm)

    t_delay_max = abs(4.14e3*max_dm*(freq[0]**-2 - freq[1]**-2))
    t_delay_max_pix = int(t_delay_max / dt)
    
    while chunksize <= t_delay_max_pix:
        chunksize *= 2
        NTIME *= 2
        print(NTIME, chunksize)

    ii=0
    params_full_arr = []

    timestr = time.strftime("%Y%m%d-%H%M%S")
    fn_fil_out = fn_out_dir + timestr + '.fil'
    params_out = fn_out_dir + timestr + '.txt'

    f_params_out = open(params_out, 'w+')
    f_params_out.write('# DM      Sigma      Time (s)     Sample    Downfact\n')

    for ii in xrange(N_FRBs):
        start, stop = chunksize*ii, chunksize*(ii+1)
        # drop FRB in random location in data chunk
        offset = int(np.random.uniform(0.1*chunksize, 0.9*chunksize)) 

        data_filobj, freq, delta_t, header = reader.read_fil_data(fn_fil, 
                                                start=start, stop=stop)

        if ii==0:
            fn_rfi_clean = reader.write_to_fil(np.zeros([NFREQ, 0]), 
                                            header, fn_fil_out)

        data = data_filobj.data
        # injected pulse time in seconds since start of file
        t0_ind = offset+NTIME//2+chunksize*ii
        t0 = t0_ind * delta_t

        if len(data)==0:
            break             

        data_event = (data[:, offset:offset+NTIME]).astype(np.float)

        data_event, params = simulate_frb.gen_simulated_frb(NFREQ=NFREQ, 
                                               NTIME=NTIME, sim=True, 
                                               fluence=5000,
                                               spec_ind=(0), width=(0.01*delta_t), 
                                               dm=dm, scat_factor=(-4, -3.5), 
                                               background_noise=data_event, 
                                               delta_t=delta_t, plot_burst=False, 
                                               freq=freq, 
                                               FREQ_REF=1400., scintillate=False)


        params.append(offset)
        print("%d/%d Injecting with DM:%d width: %.2f offset: %d" % 
                                (ii, N_FRBs, params[0], params[2], offset))
        
        data[:, offset:offset+NTIME] = data_event

        #params_full_arr.append(params)
        width = params[2]
        downsamp = max(1, int(width/delta_t))

        if rfi_clean is True:
            data = rfi_test.apply_rfi_filters(data.astype(np.float32), delta_t)

        if ii<0:
            fn_rfi_clean = reader.write_to_fil(data.transpose(), header, fn_fil_out)
        elif ii>=0:
            fil_obj = reader.filterbank.FilterbankFile(fn_fil_out, mode='readwrite')
            fil_obj.append_spectra(data.transpose())

        if calc_snr is True:
            data_filobj.dedisperse(params[0])
            data_filobj.downsample(downsamp)
            data_ts = data_filobj.data.mean(0)
            ntime = len(data_ts)
            std_chunk = scipy.signal.detrend(data_ts, type='linear')
            std_chunk.sort()
            stds = 1.148*np.sqrt((std_chunk[ntime/40:-ntime/40]**2.0).sum() /
                                   (0.95*ntime))
            snr_ = std_chunk[-1]/stds
            print(snr_)
        else:
            snr_ = 10.0
        
        f_params_out.write('%.2f %.2f %.5f %d %d\n' % (params[0], snr_, t0, t0_ind, downsamp))

        del data, data_event

    f_params_out.close()
    params_full_arr = np.array(params_full_arr)

if __name__=='__main__':
    parser = optparse.OptionParser(prog="inject_frb.py", \
                        version="", \
                        usage="%prog FN_FILTERBANK FN_FILTERBANK_OUT [OPTIONS]", \
                        description="Create diagnostic plots for individual triggers")

    parser.add_option('--sig_thresh', dest='sig_thresh', type='float', \
                        help="Only process events above >sig_thresh S/N" \
                                "(Default: 8.0)", default=8.0)

    parser.add_option('--nfrb', dest='nfrb', type='int', \
                        help="Number of FRBs to inject(Default: 50).", \
                        default=10)

    parser.add_option('--rfi_clean', dest='rfi_clean', default=False,\
                        help="apply rfi filters")

    parser.add_option('--dm_low', dest='dm_low', default=None,\
                        help="min dm to use, either float or tuple", 
                      type='float')

    parser.add_option('--dm_high', dest='dm_high', default=None,\
                        help="max dms to use, either float or tuple", 
                      type='float')


    options, args = parser.parse_args()
    fn_fil = args[0]
    fn_fil_out = args[1]
 
    if options.dm_low is None:
        if options.dm_high is None:
            dm = 500.
        else:
            dm = options.dm_high
    elif options.dm_high is None:
        dm = options.dm_low
    else:
        dm = (options.dm_low, options.dm_high)

    print("Simulating with DM:", dm)

    params = inject_in_filterbank(fn_fil, fn_fil_out, N_FRBs=options.nfrb, 
                                  NTIME=2**15, rfi_clean=options.rfi_clean, 
                                  dm=dm)

 