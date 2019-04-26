""" Tools to estimate pulsar S/N observed 
with Apertif.
"""

import numpy as np

from APERTIFparams import *


def read_atnf(name):
	""" Read in various ATNF parameters for visible 
	pulsars at WSRT
	"""
	data = np.loadtxt('./ATNF_psrs.txt',usecols=(3,4,5,8,11,14))
	names = np.genfromtxt('./ATNF_psrs.txt',usecols=(1),dtype=str)

	try:
		index = np.where(names==name)[0][0]
	except IndexError:
		print "Pulsar %s not found. Does it start with J/B?" % name 
		exit()

	RA = data[index,0]
	Dec = data[index,1]
	period = data[index,2]
	W50 = data[index,3]
	Sv400 = data[index,4]
	Sv1400 = data[index,5]

	duty_cycle = 1e-3 * W50 / period

	return period, W50, duty_cycle, Sv1400, RA, Dec

def get_psr_snr(name, tau, B=None, Ndish=1, npol=1):
	""" Get expected signoise based on 
	parameters in ATNF
	"""
	P, w, duty_cycle, S, Ra, Dec = read_atnf(name)

	# Convert mJy into Jy
	S *= 1e-3

	snr = calc_snr_psr(S, tau, duty_cycle=duty_cycle, 
		npol=npol, tel=APERTIFparams, B=B, Ndish=Ndish)

	print "Expected SNR: %f" % snr

	return snr

def calc_snr(S, tau, npol=1, tel=APERTIFparams, B=None, Ndish=1):
	""" Get snr for APERTIF single pulse

	Parameters:
	----------

	S 	 : np.float
		Flux density in Jy
	tau  : np.float
		Duration of pulse in ms
	npol : int 
		Number of polarizations
	tel  : class 
		Telescope name
	B    : np.float 
	    Bandwidth MHz
	Ndish : int
		Number of dishes
	"""
	tel = tel()

	if B is None:
		B = tel.B

	B *= 1e6
	tau /= 1000.0

	snr = S * tel.G / tel.Tsys * np.sqrt(npol*tau*B)
	snr *= Ndish

	return snr

def calc_snr_psr(S, Tint, duty_cycle=.03, npol=2, tel=APERTIFparams, B=None, Ndish=1):
	""" Get snr for APERTIF folded profile

	Parameters:
	----------

	S 	 : np.float
		Flux density in Jy
	Tint : np.float
		Total observing time
	duty_cycle : np.float 
		Duty cycle of pulsar (w / P)
	npol : int 
		Number of polarizations
	tel  : class 
		Telescope name
	B    : np.float 
	    Bandwidth MHz
	Ndish : int
		Number of dishes
	"""

	# Get total time pulse is 'on' in milliseconds
	tau = Tint / duty_cycle * 1e3

	snr = calc_snr(S, tau, tel=tel, Ndish=Ndish, B=B)

	return snr

def calc_snr_frb(S, tau, npol=2, tel=APERTIFparams, B=None, Ndish=1):
	""" Get snr for APERTIF single pulse

	Parameters:
	----------

	S 	 : np.float
		Flux density in Jy
	tau  : np.float
		Duration of pulse in ms
	npol : int 
		Number of polarizations
	tel  : class 
		Telescope name
	B    : np.float 
	    Bandwidth MHz
	Ndish : int
		Number of dishes
	"""

	snr = calc_snr(S, tau, tel=tel, Ndish=Ndish, B=B)

	return snr

if __name__=='__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('name', help='Pulsar name, e.g. B0329+54')
	parser.add_argument('time', help='Integration time [s]', type=int)
	parser.add_argument('-Ndish', help='Number of dishes', type=int, default=1)
	parser.add_argument('-npol', help='Number of polarizations', type=int, default=2)	
	parser.add_argument('-BW', help='Bandwidth used [MHz]', type=int, default=300)		
	args = parser.parse_args()

	print ""
	get_psr_snr(args.name, args.time, B=args.BW, Ndish=args.Ndish, npol=args.npol)
	print ""





