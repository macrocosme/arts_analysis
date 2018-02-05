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

import reader

# To do: 
# Put things into physical units. Scattering measure, actual widths, fluences, etc. 
# Need inputs of real telescopes. Currently it's vaguely like the Pathfinder.
# Need to not just simulate noise for the FRB triggers. 
# More single pixel widths. Unresolved bursts.
# Inverse fluence relationship right now! UPDATE DO IT 

class Event(object):
    """ Class to generate a realistic fast radio burst and 
    add the event to data. 
    """
    def __init__(self, t_ref, f_ref, dm, fluence, width, 
                 spec_ind, disp_ind=2, scat_factor=0):
        self._t_ref = t_ref
        self._f_ref = f_ref
        self._dm = dm
        self._fluence = fluence 
        self._width = width
        self._spec_ind = spec_ind
        self._disp_ind = disp_ind
        self._scat_factor = min(1, scat_factor + 1e-18) # quick bug fix hack

    def disp_delay(self, f, _dm, _disp_ind=-2.):
        """ Calculate dispersion delay in seconds for 
        frequency,f, in MHz, _dm in pc cm**-3, and 
        a dispersion index, _disp_ind. 
        """
        return 4.148808e3 * _dm * (f**(-_disp_ind))

    def arrival_time(self, f):
        t = self.disp_delay(f, self._dm, self._disp_ind)
        t = t - self.disp_delay(self._f_ref, self._dm, self._disp_ind)
        return self._t_ref + t

    def calc_width(self, dm, freq_c, bw=400.0, NFREQ=1024,
                   ti=1, tsamp=1, tau=0):
        """ Calculated effective width of pulse 
        including DM smearing, sample time, etc.
        """

        delta_freq = bw/NFREQ

        # taudm in milliseconds
        tdm = 8.3e-3 * dm * delta_freq / freq_c**3
        tI = np.sqrt(ti**2 + tsamp**2 + tdm**2 + tau**2)

        return tI

    def dm_smear(self, DM, freq_c, bw=400.0, NFREQ=1024,
                 ti=1, tsamp=0.0016, tau=0):  
        """ Calculate DM smearing SNR reduction
        """
        tau *= 1e3 # make ms
        ti *= 1e3 
        tsamp *= 1e3

        delta_freq = bw / NFREQ

        tI = np.sqrt(ti**2 + tsamp**2 + (8.3 * DM * delta_freq / freq_c**3)**2)

        return (np.sqrt(ti**2 + tau**2) / tI)**0.5

    def scintillation(self, freq):
        """ Include spectral scintillation across 
        the band. Approximate effect as a sinusoid, 
        with a random phase and a random decorrelation 
        bandwidth. 
        """
        # Make location of peaks / troughs random
        scint_phi = np.random.rand()
        # Make number of scintils between 0 and 10 (ish)
        nscint = np.exp(np.random.uniform(np.log(1e-3), np.log(10))) #liamhack change back
#        nscint = 10 #hack

        return np.cos(nscint*(freq - self._f_ref)/self._f_ref + scint_phi)**2

    def gaussian_profile(self, nt, width, t0=0.):
        """ Use a normalized Gaussian window for the pulse, 
        rather than a boxcar.
        """
        t = np.linspace(-nt//2, nt//2, nt)
        g = np.exp(-(t-t0)**2 / width**2)

        if not np.all(g > 0):
            g += 1e-18

        g /= g.max()

        return g

    def scat_profile(self, nt, f, tau=1.):
        """ Include exponential scattering profile. 
        """
        tau_nu = tau * (f / self._f_ref)**-4.
        t = np.linspace(0., nt//2, nt)

        prof = 1 / tau_nu * np.exp(-t / tau_nu)
        return prof / prof.max()

    def pulse_profile(self, nt, width, f, tau=100., t0=0.):
        """ Convolve the gaussian and scattering profiles 
        for final pulse shape at each frequency channel.
        """
        gaus_prof = self.gaussian_profile(nt, width, t0=t0)
        scat_prof = self.scat_profile(nt, f, tau) 
#        pulse_prof = np.convolve(gaus_prof, scat_prof, mode='full')[:nt]
        pulse_prof = signal.fftconvolve(gaus_prof, scat_prof)[:nt]
    
        return pulse_prof

    def add_to_data(self, delta_t, freq, data):
        """ Method to add already-dedispersed pulse 
        to background noise data. Includes frequency-dependent 
        width (smearing, scattering, etc.) and amplitude 
        (scintillation, spectral index). 
        """

        NFREQ = data.shape[0]
        NTIME = data.shape[1]
        tmid = NTIME//2

        scint_amp = self.scintillation(freq)
        rollind = 0#*int(np.random.normal(0, 5)) #hack
        
        for ii, f in enumerate(freq):
            width_ = 1e-3 * self.calc_width(self._dm, self._f_ref*1e-3, 
                                            bw=400.0, NFREQ=NFREQ,
                                            ti=self._width, tsamp=delta_t, tau=0)

#            width_ = self.dm_smear(self._dm, self._f_ref, 
#                                   delta_freq=400.0/1024, 
#                                   ti=self._width, tsamp=delta_t, tau=0)
            index_width = max(1, (np.round((width_/ delta_t))).astype(int))
            tpix = int(self.arrival_time(f) / delta_t)

            if abs(tpix) >= tmid:
                # ensure that edges of data are not crossed
                continue

            pp = self.pulse_profile(NTIME, index_width, f, 
                                    tau=self._scat_factor, t0=tpix)
            val = pp.copy()#[:len(pp)//NTIME * NTIME].reshape(NTIME, -1).mean(-1)
            val /= val.max()
            val *= self._fluence / self._width
            val = val * (f / self._f_ref) ** self._spec_ind 
            val = (0.25 + scint_amp[ii]) * val 
            val = np.roll(val, rollind)
            data[ii] += val

    def dm_transform(self, delta_t, data, freq, maxdm=5.0, NDM=50):
        """ Transform freq/time data to dm/time data.
        """
    
        if len(freq)<3:
            NFREQ = data.shape[0]
            freq = np.linspace(freq[0], freq[1], NFREQ) 

        dm = np.linspace(-maxdm, maxdm, NDM)
        ndm = len(dm)
        ntime = data.shape[-1]

        data_full = np.zeros([ndm, ntime])

        for ii, dm in enumerate(dm):
            for jj, f in enumerate(freq):
                self._dm = dm
                tpix = int(self.arrival_time(f) / delta_t)
                data_rot = np.roll(data[jj], tpix, axis=-1)
                data_full[ii] += data_rot

        return data_full

class EventSimulator():
    """Generates simulated fast radio bursts.
    Events occurrences are drawn from a Poissonian distribution.
    """

    def __init__(self, dm=(0.,2000.), fluence=(0.03,0.3),
                 width=(2*0.0016, 1.), spec_ind=(-4.,4), 
                 disp_ind=2., scat_factor=(0, 0.5), freq=(800., 400.)):
        """
        Parameters
        ----------
        datasource : datasource.DataSource object
            Source of the data, specifying the data rate and band parameters.
        dm : float or pair of floats
            Burst dispersion measure or dispersion measure range (pc cm^-2).
        fluence : float or pair of floats
            Burst fluence (at band centre) or fluence range (s).
        width : float or pair of floats.
            Burst width or width range (s).
        spec_ind : float or pair of floats.
            Burst spectral index or spectral index range.
        disp_ind : float or pair of floats.
            Burst dispersion index or dispersion index range.
        freq : tuple 
            Min and max of frequency range in MHz. Assumes low freq 
            is first freq in array, not necessarily the lowest value. 

        """

        self.width = width
        self.freq_low = freq[0]
        self.freq_up = freq[1]

        if hasattr(dm, '__iter__') and len(dm) == 2:
            self._dm = tuple(dm)
        else:
            self._dm = (float(dm), float(dm))
        if hasattr(fluence, '__iter__') and len(fluence) == 2:
            self._fluence = tuple(fluence)
        else:
            self._fluence = (float(fluence), float(fluence))
        if hasattr(width, '__iter__') and len(width) == 2:
            self._width = tuple(width)
        else:
             self._width = (float(width), float(width))
        if hasattr(spec_ind, '__iter__') and len(spec_ind) == 2:
            self._spec_ind = tuple(spec_ind)
        else:
            self._spec_ind = (float(spec_ind), float(spec_ind))
        if hasattr(disp_ind, '__iter__') and len(disp_ind) == 2:
            self._disp_ind = tuple(disp_ind)
        else:
            self._disp_ind = (float(disp_ind), float(disp_ind))
        if hasattr(scat_factor, '__iter__') and len(scat_factor) == 2:
            self._scat_factor = tuple(scat_factor)
        else:
            self._scat_factor = (float(scat_factor), float(scat_factor))

        # self._freq = datasource.freq
        # self._delta_t = datasource.delta_t

        self._freq = np.linspace(self.freq_low, self.freq_up, 256) # tel parameter 

    def draw_event_parameters(self):
        dm = uniform_range(*self._dm)
        fluence = uniform_range(*self._fluence)**(-2/3.)/0.5**(-2/3.)
        spec_ind = uniform_range(*self._spec_ind)
        disp_ind = uniform_range(*self._disp_ind)
        # turn this into a log uniform dist. Note not *that* many 
        # FRBs have been significantly scattered. Should maybe turn this 
        # knob down.
        scat_factor = np.exp(np.random.uniform(*self._scat_factor))
        # change width from uniform to lognormal
        width = np.random.lognormal(np.log(self._width[0]), self._width[1])
        width = max(min(width, 100*self._width[0]), 0.5*self._width[0])
        return dm, fluence, width, spec_ind, disp_ind, scat_factor

def uniform_range(min_, max_):
    return random.uniform(min_, max_)


def gen_simulated_frb(NFREQ=16, NTIME=250, sim=True, fluence=(0.03,0.3),
                spec_ind=(-4, 4), width=(2*0.0016, 1), dm=(-0.01, 0.01),
                scat_factor=(-3, -0.5), background_noise=None, delta_t=0.0016,
                plot_burst=False, freq=(800, 400), FREQ_REF=600., 
                ):
    """ Simulate fast radio bursts using the EventSimulator class.

    Parameters
    ----------
    NFREQ       : np.int 
        number of frequencies for simulated array
    NTIME       : np.int 
        number of times for simulated array
    sim         : bool 
        whether or not to simulate FRB or just create noise array
    spec_ind    : tuple 
        range of spectral index 
    width       : tuple 
        range of widths in seconds (atm assumed dt=0.0016)
    scat_factor : tuple 
        range of scattering measure (atm arbitrary units)
    background_noise : 
        if None, simulates white noise. Otherwise should be an array (NFREQ, NTIME)
    plot_burst : bool 
        generates a plot of the simulated burst

    Returns
    -------
    data : np.array 
        data array (NFREQ, NTIME)
    parameters : tuple 
        [dm, fluence, width, spec_ind, disp_ind, scat_factor]

    """
    plot_burst = False

    # Hard code incoherent Pathfinder data time resolution
    # Maybe instead this should take a telescope class, which 
    # has all of these things already.
    t_ref = 0. # hack

    if len(freq) < 3:
        freq=np.linspace(freq[0], freq[1], NFREQ)      

    if background_noise is None:
        # Generate background noise with unit variance
        data = np.random.normal(0, 1, NTIME*NFREQ).reshape(NFREQ, NTIME)
    else:
        data = background_noise

    # What about reading in noisy background?
    if sim is False:
        return data, []

    # Call class using parameter ranges
    ES = EventSimulator(dm=dm, scat_factor=scat_factor, fluence=fluence, 
                        width=width, spec_ind=spec_ind)
    # Realize event parameters for a single FRB
    dm, fluence, width, spec_ind, disp_ind, scat_factor = ES.draw_event_parameters()
    # Create event class with those parameters 
    E = Event(t_ref, FREQ_REF, dm, 10e-4*fluence, 
              width, spec_ind, disp_ind, scat_factor)
    # Add FRB to data array 
    E.add_to_data(delta_t, freq, data)

    if plot_burst:
        subplot(211)
        imshow(data.reshape(-1, NTIME), aspect='auto', 
               interpolation='nearest', vmin=0, vmax=10)
        subplot(313)
        plot(data.reshape(-1, ntime).mean(0))

    return data, [dm, fluence, width, spec_ind, disp_ind, scat_factor]


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


def inject_in_filterbank(fn_fil, fn_fil_out, N_FRBs=1, 
                         NFREQ=1536, NTIME=2**15):
    """ Inject an FRB in each chunk of data 
        at random times. Default params are for Apertif data.
    """

    chunksize = 5e5
    ii=0

    params_full_arr = []

    for ii in xrange(N_FRBs):
        start, stop = chunksize*ii, chunksize*(ii+1)
        # drop FRB in random location in data chunk
        offset = int(np.random.uniform(0.1*chunksize, 0.9*chunksize)) 
        start=1000000000
        data, freq, delta_t, header = reader.read_fil_data(fn_fil, 
                                                start=start, stop=stop)

        # injected pulse time in seconds since start of file
        t0_ind = offset+NTIME//2+chunksize*ii
        t0 = t0_ind * delta_t

        if data==0:
            break             

        data_event = (data[offset:offset+NTIME].transpose()).astype(np.float)

        data_event, params = gen_simulated_frb(NFREQ=NFREQ, 
                                               NTIME=NTIME, sim=True, fluence=(0.01, 1.), 
                                               spec_ind=(-4, 4), width=(delta_t, 2), 
                                               dm=(100, 1000), scat_factor=(-4, -0.5), 
                                               background_noise=data_event, 
                                               delta_t=delta_t, plot_burst=False, 
                                               freq=(1550, 1250), 
                                               FREQ_REF=1550.)

        params.append(offset)
        print("Injecting with DM:%f width: %f offset: %d" % 
                                (params[0], params[2], offset))
        
        data[offset:offset+NTIME] = data_event.transpose()

        #params_full_arr.append(params)
        width = params[2]
        downsamp = max(1, int(width/delta_t))

        params_full_arr.append([params[0], 20.0, t0, t0_ind, downsamp])

        if ii==0:
            fn_rfi_clean = reader.write_to_fil(data, header, fn_fil_out)
        elif ii>0:
            fil_obj = reader.filterbank.FilterbankFile(fn_fil_out, mode='readwrite')
            fil_obj.append_spectra(data) 

        del data 

    params_full_arr = np.array(params_full_arr)

    np.savetxt('/home/arts/connor/arts-analysis/simulated.singlepulse', params_full_arr)

    return params_full_arr

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









