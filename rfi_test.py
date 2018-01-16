import numpy as np

import filterbank
import preprocess

HPF_WIDTH = 100.0 * 1e-3 # HPF width in seconds

def write_to_fil(data, header, fn):
	fn_rfi_clean = fn.split('.fil')[0] + '_rfi_clean.fil'
	filterbank.create_filterbank_file(
		fn_rfi_clean, header, spectra=data, mode='readwrite')
	print "Writing to %s" % fn_rfi_clean

	return fn_rfi_clean

def get_freqs(fil_obj):
	fch1 = fil_obj.header['fch1']
	foff = fil_obj.header['foff']
	nchans = fil_obj.header['nchans']

	freq = np.linspace(fch1, fch1+nchans*foff, nchans)
	return freq

def dedisperse_timestream(data, freq, dm, delta_t):
	""" data (ntime, nfreq)
	"""
	fref = 1400.
	nfreq = len(data)

	for fi in xrange(nfreq):         
		tau = dm_delays(dm, freq[fi], fref)
		taubin = int(round(tau / delta_t))
		data[fi] = np.roll(data[fi], -taubin)

	return data

def dm_delays(dm, freq, f_ref):
	""" freq in MHz
	"""
	return 4.148808e3 * dm * (freq**(-2) - f_ref**(-2))

def plotsome(data):
	figure(figsize=(12,12))

	subplot(311)
	vm = 5 * np.std(data)
	data -= data.mean(-1)[:, None]
	imshow(data, aspect='auto', interpolation='nearest',vmax=vm,vmin=-vm,cmap='Greys')

	subplot(312)
	plot(data.mean(0))

	subplot(313)
	plot(data.mean(-1))

	show()

def apply_rfi_filters(data, delta_t):
	print "Applying RFI filters"
	data = preprocess.highpass_filter(data, HPF_WIDTH / delta_t)
	print "     highpass_filter"
	preprocess.remove_outliers(data, 5)
	print "     remove_outliers"
	preprocess.remove_noisy_freq(data, 3)
	preprocess.remove_bad_times(data, 3)
	preprocess.remove_noisy_freq(data, 3)	
	print "     time/freq cuts"

	return data

def read_fil_data(fn, start=0, stop=1e7):
	print "Reading filterbank file %s \n" % fn
	fil_obj = filterbank.FilterbankFile(fn)
	header = fil_obj.header
	delta_t = fil_obj.header['tsamp'] # delta_t in milliseconds
	freq = get_freqs(fil_obj)
	data = fil_obj.get_spectra(start, stop)
	# turn array into time-major, for preprocess
#	data = data.transpose() 

	return data, freq, delta_t, header

def process_full_fil(fn):
	ii=0
	while True:
		start, stop = 5e5*ii, 5e5*(ii+1)
		data, freq, delta_t, header = read_fil_data(fn, start=start, stop=stop)

		if len(data[0])==0:
			break

		print data.shape, start
#		data = dataapply_rfi_filters(data, delta_t)

		if ii==0:
			fn_rfi_clean = write_to_fil(data.transpose(), header, fn)
		elif ii>0:
			fil_obj = filterbank.FilterbankFile(fn_rfi_clean, mode='readwrite')
			fil_obj.append_spectra(data.transpose())

		ii+=1

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('filename', help='name of filterbank file')



