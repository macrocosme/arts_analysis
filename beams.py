import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit




class BeamFits:

    def __init__(self, dt=25):
        self.dt = dt
        self.times = None
        self.nfreq = 1536
        self.fch1 = 1550.0
        self.foff = -0.1953125
        self.freq = np.linspace(self.fch1, 
                        self.fch1 + self.foff*self.nfreq, self.nfreq)


    @property
    def CB_map(self):
        CB_map = np.array([[3, 8, 13, 18],
                   [0, 1, 2, 5],
                   [12, 15, 16, 17],
                   [20, 21, 22, 25],
                   [26, 27, 30, 21],
                   [32, 35, 36, 37],     
                   [23, 28, 33, 38]])

        return CB_map

    def get_times(self, ntime):
        self.times = np.linspace(0, self.dt*ntime, ntime)
        return self.times

    def sigma_to_beamwidth(self, sig_sec, dec=0):
        FWHM_sec = 2.355*sig_sec
        FWHM_deg = FWHM_sec / 86400. * 360 * np.cos(np.deg2rad(dec))
        return FWHM_deg

    def gaussian(self, x, mu, sig, A=1, B=0):
        return A * np.exp(-(x - mu)**2 / (2*sig**2)) + B

    def fit_gaussian(self, x, y, params_0):
        """ params_0 = [mu, sig, A, B] 
            e.g. [0, 0.25, 1, 0]
        """
        popt, pcov = curve_fit(self.gaussian, x, y, p0=params_0, maxfev=10000)

        return popt, pcov

    def get_beam_number(self, fn):
        return int(fn.split('CB')[-1][:2])

    def read_in_beams(self, flist):

        data_full = []
        CB_number = []

        nbeams = len(flist)

        for fn in flist:
            data = np.load(fn)
            data = data.reshape(-1, nfreq)
            data_full.append(data)

            CB_number.append(self.get_beam_number(fn))

        data_full = np.concatenate(data_full)
        data_full = data_full.reshape((nbeams, ) + data.shape)

        return data_full, np.array(CB_number)

    def fit_all_beams(self, input_data, freq=1110):
        if type(input_data) == list:
            data_full = self.read_in_beams(input_data)[0]
        elif type(input_data) == numpy.ndarray:
            data_full = input_data

        ntime = data_full.shape[1]
        times = self.get_times(ntime)
        
        popt_full = []

        for data in data_full[..., freq]:
            B0 = np.median(data)
            A0 = data.max() - B0
            mu0 = times[np.argmax(data)]
            sig0 = 200
            params_0 = [mu0, sig0, A0, B0]
            print "Guess:", np.round(params_0)
            try:
                popt = self.fit_gaussian(times, data, params_0)[0]
                print "Fit: ", np.round(popt)
                popt_full.append(popt)
            except:
                popt_full.append(params_0)

        return popt_full

    def plot_all(self, flist, popt=None):
        data_full = self.read_in_beams(flist)[0]

        legend_str = []

        fig = plt.figure()

        for ii, data in enumerate(data_full):
            ntime = data.shape[0]
            times = self.get_times(ntime)
            plt.plot(times, data[:, 1110])
            nm = flist[ii].split('/')[-1][2:4]
            legend_str.append(nm)
        
        if popt is not None:
            for pp in popt:
                gauss = self.gaussian(times, pp[0], pp[1], A=pp[2], B=pp[3])
                plt.plot(times, gauss,'--')
            #nm = flist[ii].split('/')[-1][2:4]
            #legend_str.append(nm)

        plt.legend(legend_str)
        plt.show()            

    def plot_PAF_config(self, data_full, CB_list, 
                        freq=1110, twoD=False):

        fig = plt.figure()
        for ii, cb in enumerate(CB_list):
            x, y = np.where(self.CB_map==cb)
            try:
                plt.subplot(7, 4, 4*x[0]+y[0]+1)
                print data_full.shape
                plt.plot(times, data_full[ii, :, freq])
            except:
                continue

        plt.show()

        if twoD is False:
            return

        fig = plt.figure()
        for ii, cb in enumerate(CB_list):
            x, y = np.where(self.CB_map==cb)
            try:
                plt.subplot(7, 4, 4*x[0]+y[0]+1)
                dd = data_full[ii, :, freq]
                plt.imshow(dd[None]*dd[:, None], cmap='RdBu')
                plt.axis('off')
            except:
                continue

        plt.show()

    def fit_all_freq(self):

        params_full = []
        for ff in range(nfreq):
            p = B.fit_all_beams(fl, freq=ff)
            p = np.array(p)
            params_full.append(p)

        params_full = np.concatenate(params_full)

        return params_full

for ii, cb in enumerate(CB_list):
    x, y = np.where(CB_map==cb)
    try:
        ind = 4*x[0]+y[0]+1
        subplot(7,4,ind)
        if ind<5 or ind>24:
            col='red'
        else:
            col='blue'
        fill_between(times, 0, 1000, color=col, alpha=0.25)
        plot(times, d[ii, :, 1110], color='black')
        xlim(times[0], times[-1])
        ylim(0.9*d[ii,:,1110].min(), 1.1*d[ii, :, 1110].max())
        axis('off')
        legend(['CB'+str(cb)], loc=2)
    except:
        pass

for ii, cb in enumerate(CB_list):
    x, y = np.where(CB_map==cb)
    try:
        ind = 4*x[0]+y[0]+1
        subplot(7,4,ind)
        if ind<5 or ind>24:
            col='red'
        else:
            col='blue'
        # fill_between(times, 0, 1000, color=col, alpha=0.25)
        dd = d[ii,:,1110]
        imshow(dd[None]*dd[:,None])
        # xlim(times[0], times[-1])
        # ylim(0.9*d[ii,:,1110].min(), 1.1*d[ii, :, 1110].max())
        axis('off')
        #legend(['CB'+str(cb)], loc=2)
    except:
        pass

def plot_all(flist, dt=25):
    legend_str = []

    fig = plt.figure()

    for fn in flist:
        data = np.load(fn)
        data = data.reshape(-1, nfreq)
        ntime = data.shape[0]
        times = np.linspace(0, dt*ntime, ntime)
        plt.plot(times, data[:, 1110])
        nm = fn.split('/')[-1][2:4]
        legend_str.append(nm)
    
    plt.legend(legend_str)
    plt.show()
