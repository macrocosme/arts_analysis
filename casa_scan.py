import numpy as np
import matplotlib.pylab as plt

import psrchive
import glob

def casa_flux(nu_MHz):
    return 5000.0 * (nu_MHz / 340.0)**-0.804
    
flist = glob.glob('/data/15/Timing/20170302/2017.03.02-10:35:30.B0000+00/2017.03.02-10:35:30.B0000+00.band14_0*.ar') 
flist.sort()

data_arr = []

for ff in flist[:]:
    arch = psrchive.Archive_load(ff)
    data = arch.get_data()
    data = data.sum(axis=-1)
    data_arr.append(data)

nfreq = data.shape[-1]
cfreq = arch.get_centre_frequency()
bw = 131.25
freq = np.linspace(cfreq, cfreq + bw, nfreq) - bw/2.
print freq

data_arr = np.concatenate(data_arr, axis=0)
print data_arr.shape

fig = plt.figure(figsize=(15, 12))

G = (26 / 64.)**2*0.7

print G

for i in range(12):
    fig.add_subplot(3,4,i+1)
    data = data_arr[:, 0, i:i+2].sum(-1)
    data /= np.median(data[140:])
    Snu = casa_flux(freq[2*i])

    plt.plot(G * Snu / (data - 1), '.', lw=3, color='black')
    plt.ylim(0, 1.2e2)
    plt.legend([str(np.round(freq[2*i]))+'MHz'])
    plt.axhline(75.0, linestyle='--', color='red')
    plt.xlim(40, 140)

    if i % 4 == 0:
        plt.ylabel(r'$T_{sys}$', fontsize=20)

plt.show()
plt.savefig('here.png')

