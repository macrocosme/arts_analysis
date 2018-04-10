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
import rfi_test

# To do: 
# Put things into physical units. Scattering measure, actual widths, fluences, etc. 
# Need inputs of real telescopes. Currently it's vaguely like the Pathfinder.
# Need to not just simulate noise for the FRB triggers. 
# More single pixel widths. Unresolved bursts.
# Inverse fluence relationship right now! UPDATE DO IT 

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
                         NFREQ=1536, NTIME=2**15):
    """ Inject an FRB in each chunk of data 
        at random times. Default params are for Apertif data.
    """

    chunksize = 5e4
    ii=0

    params_full_arr = []

    timestr = time.strftime("%Y%m%d-%H%M%S")
    fn_fil_out = fn_out_dir + timestr + '.fil'
    params_out = fn_out_dir + timestr + '.txt'

    for ii in xrange(N_FRBs):
        start, stop = chunksize*ii, chunksize*(ii+1)
        # drop FRB in random location in data chunk
        offset = int(np.random.uniform(0.1*chunksize, 0.9*chunksize)) 

        data, freq, delta_t, header = reader.read_fil_data(fn_fil, 
                                                start=start, stop=stop)

        if ii==0:
            fn_rfi_clean = reader.write_to_fil(np.zeros([NFREQ, 0]), header, fn_fil_out)

        data = data.data
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
                                               dm=(1000.0), scat_factor=(-4, -3.5), 
                                               background_noise=data_event, 
                                               delta_t=delta_t, plot_burst=False, 
                                               freq=(1550, 1250), 
                                               FREQ_REF=1400., scintillate=False)


        params.append(offset)
        print("Injecting with DM:%f width: %f offset: %d" % 
                                (params[0], params[2], offset))
        
        data[:, offset:offset+NTIME] = data_event

        #params_full_arr.append(params)
        width = params[2]
        downsamp = max(1, int(width/delta_t))

        params_full_arr.append([params[0], 20.0, t0, t0_ind, downsamp])

        if rfi_clean is True:
            data = rfi_test.apply_rfi_filters(data, delta_t)

        if ii<0:
            fn_rfi_clean = reader.write_to_fil(data.transpose(), header, fn_fil_out)
        elif ii>=0:
            fil_obj = reader.filterbank.FilterbankFile(fn_fil_out, mode='readwrite')
            fil_obj.append_spectra(data.transpose())

    params_full_arr = np.array(params_full_arr)
    np.savetxt(params_out, params_full_arr)

    return data_event, data

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


    options, args = parser.parse_args()
    fn_fil = args[0]
    fn_fil_out = args[1]

    params = inject_in_filterbank(fn_fil, fn_fil_out, N_FRBs=options.nfrb, 
                                  NTIME=2**15)

